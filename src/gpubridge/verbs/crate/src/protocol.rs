//! The verbs bridge's wire protocol, version 2 (docs/VERBS_TRANSPORT.md, "Wire
//! protocol"): the message header, the handshake text the members exchange through the
//! c10d store, and how a member picks its device, port and GID.
//!
//! Every message is one RC SEND of at most `slot` bytes: a [`HEADER_BYTES`]-byte header,
//! then up to `slot - HEADER_BYTES` bytes of payload. With rendezvous on (`rndv` bytes in
//! the handshake, the same on every member), a message of at least `rndv` bytes waits for
//! the receiver's clear-to-send: either a write grant (address, length and the remote key
//! of memory registered for that one receive), after which the sender RDMA WRITEs the
//! payload there and SENDs a "written" message, or "send it eagerly", after which it goes
//! in chunks like any other. Remote keys cross the fabric only in write grants; they are
//! never logged, printed or reported.

use std::fmt::Write as _;

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::resources::Context;
use crate::types::{Gid, GidType, LinkLayer, Mtu, PortAttr, PortState};

/// Bumped whenever the header, the handshake or the rules of either change.
pub const PROTOCOL_VERSION: u32 = 2;
pub const HEADER_BYTES: usize = 48;
/// "VB", little-endian, at the start of every header.
const MAGIC: u16 = 0x4256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A chunk of a tagged message.
    Data,
    /// Returns credits only (the header's counters); carries no payload.
    Credit,
    /// The sender is closing: it sends nothing after this.
    Bye,
    /// Clear to send, by RDMA WRITE: the receiver posted a receive for `tag` of
    /// `msg_len` bytes into memory at `offset` (an address) that `key` grants writes to.
    Cts,
    /// Clear to send, in chunks: the receiver posted a receive for `tag` of `msg_len`
    /// bytes into memory the peer can't write.
    Go,
    /// The payload of `tag` (`msg_len` bytes) has been written into the granted memory.
    Fin,
}

impl Kind {
    fn code(self) -> u8 {
        match self {
            Kind::Data => 1,
            Kind::Credit => 2,
            Kind::Bye => 3,
            Kind::Cts => 4,
            Kind::Go => 5,
            Kind::Fin => 6,
        }
    }

    fn from_code(code: u8) -> Option<Kind> {
        Some(match code {
            1 => Kind::Data,
            2 => Kind::Credit,
            3 => Kind::Bye,
            4 => Kind::Cts,
            5 => Kind::Go,
            6 => Kind::Fin,
            _ => return None,
        })
    }

    /// Whether it takes one of the peer's data receives (and so a data credit): data and
    /// the rendezvous messages. Credit and goodbye messages take control receives.
    pub fn data_class(self) -> bool {
        !matches!(self, Kind::Credit | Kind::Bye)
    }
}

/// A message header. All integers little-endian:
///
/// | bytes | field |
/// | --- | --- |
/// | 0-1 | magic `0x4256` |
/// | 2 | protocol version |
/// | 3 | kind (1 data, 2 credit, 3 bye, 4 clear to write, 5 clear to send, 6 written) |
/// | 4-7 | sequence number of this message on this queue pair (from 0, wrapping) |
/// | 8-11 | payload bytes in this message |
/// | 12-15 | data receives the sender has reposted for its peer, cumulative (wrapping) |
/// | 16-19 | control receives the sender has reposted for its peer, cumulative (wrapping) |
/// | 20-23 | clear to write: the remote key of the granted memory; else zero |
/// | 24-31 | tag |
/// | 32-39 | the whole message's length in bytes |
/// | 40-47 | data: this chunk's offset in the message; clear to write: the address |
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: Kind,
    pub seq: u32,
    pub chunk_len: u32,
    pub data_freed: u32,
    pub ctrl_freed: u32,
    /// A write grant's remote key (clear to write only). Never printed: `Debug` leaves
    /// it out.
    pub key: u32,
    pub tag: u64,
    pub msg_len: u64,
    pub offset: u64,
}

impl std::fmt::Debug for Header {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Header")
            .field("kind", &self.kind)
            .field("seq", &self.seq)
            .field("chunk_len", &self.chunk_len)
            .field("data_freed", &self.data_freed)
            .field("ctrl_freed", &self.ctrl_freed)
            .field("tag", &self.tag)
            .field("msg_len", &self.msg_len)
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

impl Header {
    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut b = [0u8; HEADER_BYTES];
        b[0..2].copy_from_slice(&MAGIC.to_le_bytes());
        b[2] = PROTOCOL_VERSION as u8;
        b[3] = self.kind.code();
        b[4..8].copy_from_slice(&self.seq.to_le_bytes());
        b[8..12].copy_from_slice(&self.chunk_len.to_le_bytes());
        b[12..16].copy_from_slice(&self.data_freed.to_le_bytes());
        b[16..20].copy_from_slice(&self.ctrl_freed.to_le_bytes());
        b[20..24].copy_from_slice(&self.key.to_le_bytes());
        b[24..32].copy_from_slice(&self.tag.to_le_bytes());
        b[32..40].copy_from_slice(&self.msg_len.to_le_bytes());
        b[40..48].copy_from_slice(&self.offset.to_le_bytes());
        b
    }

    /// The header of a message of `received` bytes (the completion's length).
    pub fn decode(b: &[u8; HEADER_BYTES], received: u32) -> std::result::Result<Header, String> {
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        if u16::from_le_bytes([b[0], b[1]]) != MAGIC {
            return Err("a message without the protocol's magic number".into());
        }
        if u32::from(b[2]) != PROTOCOL_VERSION {
            return Err(format!(
                "a message of protocol version {}, not {PROTOCOL_VERSION}",
                b[2]
            ));
        }
        let kind = Kind::from_code(b[3]).ok_or_else(|| format!("unknown message kind {}", b[3]))?;
        let h = Header {
            kind,
            seq: u32_at(4),
            chunk_len: u32_at(8),
            data_freed: u32_at(12),
            ctrl_freed: u32_at(16),
            key: u32_at(20),
            tag: u64_at(24),
            msg_len: u64_at(32),
            offset: u64_at(40),
        };
        if u64::from(received) != HEADER_BYTES as u64 + u64::from(h.chunk_len) {
            return Err(format!(
                "a message of {received} bytes whose header says {} of payload",
                h.chunk_len
            ));
        }
        if kind != Kind::Data && h.chunk_len != 0 {
            return Err("a control message with a payload".into());
        }
        if kind != Kind::Cts && h.key != 0 {
            return Err("a remote key outside a write grant".into());
        }
        if kind == Kind::Cts && h.offset == 0 {
            return Err("a write grant without an address".into());
        }
        if matches!(kind, Kind::Cts | Kind::Go | Kind::Fin) && h.msg_len == 0 {
            return Err("a rendezvous for an empty message".into());
        }
        let end = h.offset.checked_add(u64::from(h.chunk_len));
        if kind == Kind::Data && end.map_or(true, |e| e > h.msg_len) {
            return Err(format!(
                "a chunk of {} bytes at {} of a {}-byte message",
                h.chunk_len, h.offset, h.msg_len
            ));
        }
        Ok(h)
    }
}

/// What one member tells the others in the handshake (one line of text through the c10d
/// store). Addresses and queue pair numbers only: there are no remote keys in this
/// protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    pub version: u32,
    pub link_layer: LinkLayer,
    pub lid: u16,
    /// The GID this member sends from (RoCE), or its port's GID 0 (InfiniBand).
    pub gid: Gid,
    /// Whether this member addresses its peers with a global route (always on RoCE; on
    /// InfiniBand only when a GID index was set).
    pub grh: bool,
    pub active_mtu: Mtu,
    pub slot_bytes: u32,
    pub credits: u32,
    pub ctrl_credits: u32,
    /// Messages of at least this many bytes go by rendezvous; 0: none do.
    pub rndv: u64,
    /// The queue pair number this member uses for each member, by member index (0 for
    /// itself).
    pub qpns: Vec<u32>,
    /// The first PSN it sends with on each of them.
    pub psns: Vec<u32>,
    /// For error messages and telemetry.
    pub device: String,
    pub port: u8,
}

fn link_name(l: LinkLayer) -> &'static str {
    match l {
        LinkLayer::InfiniBand => "ib",
        LinkLayer::Ethernet => "eth",
    }
}

pub fn link_text(l: LinkLayer) -> &'static str {
    match l {
        LinkLayer::InfiniBand => "InfiniBand",
        LinkLayer::Ethernet => "Ethernet (RoCE)",
    }
}

pub fn gid_hex(gid: &Gid) -> String {
    gid.0.iter().fold(String::with_capacity(32), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn parse_gid(text: &str) -> Option<Gid> {
    if text.len() != 32 {
        return None;
    }
    let mut g = [0u8; 16];
    for (i, byte) in g.iter_mut().enumerate() {
        *byte = u8::from_str_radix(text.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(Gid(g))
}

fn list(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

impl Info {
    /// `gbverbs <version>;key=value;...`. Readers ignore keys they don't know.
    pub fn encode(&self) -> String {
        format!(
            "gbverbs {};ll={};lid={};gid={};grh={};mtu={};slot={};credits={};ctrl={};rndv={};\
             qpn={};psn={};dev={};port={}",
            self.version,
            link_name(self.link_layer),
            self.lid,
            gid_hex(&self.gid),
            u8::from(self.grh),
            self.active_mtu.bytes(),
            self.slot_bytes,
            self.credits,
            self.ctrl_credits,
            self.rndv,
            list(&self.qpns),
            list(&self.psns),
            self.device,
            self.port
        )
    }

    pub fn decode(text: &str) -> std::result::Result<Info, String> {
        let mut parts = text.trim().split(';');
        let version = parts
            .next()
            .and_then(|p| p.strip_prefix("gbverbs "))
            .and_then(|v| v.parse::<u32>().ok())
            .ok_or_else(|| format!("not a verbs bridge handshake: {text:?}"))?;
        if version != PROTOCOL_VERSION {
            // Nothing else can be trusted to mean the same.
            return Err(format!(
                "it speaks verbs bridge protocol {version}, this side {PROTOCOL_VERSION}; \
                 use the same gpubridge on every leader"
            ));
        }
        let mut fields = std::collections::HashMap::new();
        for p in parts {
            if let Some((k, v)) = p.split_once('=') {
                fields.insert(k, v);
            }
        }
        let get = |k: &str| {
            fields
                .get(k)
                .copied()
                .ok_or_else(|| format!("handshake without {k}"))
        };
        let num = |k: &str| -> std::result::Result<u64, String> {
            get(k)?
                .parse::<u64>()
                .map_err(|_| format!("handshake {k} isn't a number"))
        };
        let nums = |k: &str| -> std::result::Result<Vec<u32>, String> {
            get(k)?
                .split(',')
                .map(|v| v.parse::<u32>().map_err(|_| format!("bad handshake {k}")))
                .collect()
        };
        let small = |k: &str, max: u64| -> std::result::Result<u64, String> {
            let v = num(k)?;
            if v > max {
                return Err(format!("handshake {k} {v} out of range"));
            }
            Ok(v)
        };
        let link_layer = match get("ll")? {
            "ib" => LinkLayer::InfiniBand,
            "eth" => LinkLayer::Ethernet,
            other => return Err(format!("unknown link layer {other:?}")),
        };
        Ok(Info {
            version,
            link_layer,
            lid: small("lid", u64::from(u16::MAX))? as u16,
            gid: parse_gid(get("gid")?).ok_or("bad handshake gid")?,
            grh: small("grh", 1)? == 1,
            active_mtu: Mtu::from_bytes(small("mtu", 4096)? as u32).ok_or("bad handshake mtu")?,
            slot_bytes: small("slot", u64::from(u32::MAX))? as u32,
            credits: small("credits", u64::from(u32::MAX))? as u32,
            ctrl_credits: small("ctrl", u64::from(u32::MAX))? as u32,
            rndv: num("rndv")?,
            qpns: nums("qpn")?,
            psns: nums("psn")?,
            device: get("dev")?.to_owned(),
            port: small("port", 255)? as u8,
        })
    }

    /// Checks that a member that sent `theirs` can talk to this one, and returns the path
    /// MTU to use: the smaller active MTU. `member` and `index` are their and our member
    /// indices.
    pub fn compatible(&self, theirs: &Info, member: usize, index: usize) -> Result<Mtu> {
        let here = format!("{} port {}", self.device, self.port);
        let there = format!("{} port {} on member {member}", theirs.device, theirs.port);
        if theirs.link_layer != self.link_layer {
            return Err(Error::invalid(format!(
                "incompatible link layers: {} here ({here}), {} there ({there}); a reliable \
                 connection needs the same fabric on both ends",
                link_text(self.link_layer),
                link_text(theirs.link_layer)
            )));
        }
        let settings = |i: &Info| (i.slot_bytes, i.credits, i.ctrl_credits, i.rndv);
        if settings(theirs) != settings(self) {
            return Err(Error::invalid(format!(
                "message slot, credits, control credits and rendezvous bytes are {:?} here but \
                 {:?} on member {member}; use the same settings on every leader",
                settings(self),
                settings(theirs)
            )));
        }
        if theirs.qpns.len() != self.qpns.len() || theirs.psns.len() != self.qpns.len() {
            return Err(Error::invalid(format!(
                "member {member} has {} members in its handshake, this side {}",
                theirs.qpns.len(),
                self.qpns.len()
            )));
        }
        match self.link_layer {
            LinkLayer::InfiniBand if theirs.lid == 0 || theirs.lid >= 0xC000 => {
                return Err(Error::invalid(format!(
                    "{there} has no unicast LID ({:#x}): is a subnet manager running?",
                    theirs.lid
                )));
            }
            _ => {}
        }
        if (self.link_layer == LinkLayer::Ethernet || self.grh) && theirs.gid.is_zero() {
            return Err(Error::invalid(format!(
                "{there} sent no GID, and this side addresses peers by GID"
            )));
        }
        let qpn = theirs.qpns[index];
        if !(2..1 << 24).contains(&qpn) || theirs.psns[index] >= 1 << 24 {
            return Err(Error::invalid(format!(
                "member {member}'s queue pair number or PSN for this member is invalid"
            )));
        }
        Ok(self.active_mtu.min(theirs.active_mtu))
    }
}

/// The device, port and GID a member uses.
pub struct Selected {
    pub ctx: Context,
    pub port: u8,
    pub attr: PortAttr,
    /// The GID index this member sends from (RoCE; InfiniBand when set explicitly).
    pub gid_index: Option<u32>,
    pub gid: Gid,
    pub gid_type: Option<GidType>,
}

/// How good a GID is for automatic choice on a RoCE port (lower is better), or None if
/// it can't be used.
fn gid_rank(gid: &Gid, kind: GidType) -> Option<u8> {
    if gid.is_zero() {
        return None;
    }
    let v4_mapped = gid.0[..10] == [0; 10] && gid.0[10..12] == [0xff, 0xff];
    let link_local = gid.0[0] == 0xfe && gid.0[1] & 0xc0 == 0x80;
    match kind {
        GidType::RoceV2 if v4_mapped => Some(0),
        GidType::RoceV2 if !link_local => Some(1),
        GidType::RoceV2 => Some(2),
        GidType::RoceV1 => Some(3),
        _ => None,
    }
}

/// Why a port can't be used, or None.
fn port_problem(attr: &PortAttr) -> Option<String> {
    if attr.state != PortState::Active {
        return Some(format!("{:?}, not Active", attr.state));
    }
    if attr.link_layer == LinkLayer::InfiniBand && (attr.lid == 0 || attr.lid >= 0xC000) {
        return Some(format!(
            "no unicast LID ({:#x}): is a subnet manager running?",
            attr.lid
        ));
    }
    None
}

/// The GID index to send from on a RoCE port: `wanted` if given (it must hold a GID),
/// else the best by [`gid_rank`] (the lowest index among equals).
fn pick_gid(
    ctx: &Context,
    port: u8,
    attr: &PortAttr,
    wanted: Option<u32>,
) -> std::result::Result<(u32, Gid, GidType), String> {
    if let Some(i) = wanted {
        let e = ctx
            .query_gid(port, i)
            .map_err(|e| format!("GID index {i}: {e}"))?;
        if e.gid.is_zero() {
            return Err(format!("GID index {i} is empty"));
        }
        return Ok((i, e.gid, e.gid_type));
    }
    let mut best: Option<(u8, u32, Gid, GidType)> = None;
    for i in 0..attr.gid_tbl_len {
        // Unpopulated entries fail to query on some providers: skip them.
        let Ok(e) = ctx.query_gid(port, i) else {
            continue;
        };
        if let Some(rank) = gid_rank(&e.gid, e.gid_type) {
            if best.as_ref().map_or(true, |b| rank < b.0) {
                best = Some((rank, i, e.gid, e.gid_type));
            }
        }
    }
    best.map(|(_, i, g, t)| (i, g, t))
        .ok_or_else(|| "no usable RoCE GID (set GPUBRIDGE_VERBS_GID_INDEX)".to_string())
}

/// Picks the device, port and GID (docs/VERBS_TRANSPORT.md, "Choosing the endpoint"):
/// the device, port and GID index given, where given; otherwise devices in name order,
/// their ports in order, the first Active port with a usable address, and on RoCE the
/// best GID by [`gid_rank`]. An explicitly named device or port is never swapped for
/// another one: if it can't be used, this fails and says why.
pub fn select(
    provider: &Arc<dyn Provider>,
    device: Option<&str>,
    port: Option<u8>,
    gid_index: Option<u32>,
) -> Result<Selected> {
    let mut devices = provider.list_devices()?;
    if devices.is_empty() {
        return Err(Error::Unsupported(format!(
            "{} lists no RDMA devices",
            provider.name()
        )));
    }
    devices.sort();
    let candidates: Vec<String> = match device {
        Some(d) if !devices.iter().any(|x| x == d) => {
            return Err(Error::invalid(format!(
                "no RDMA device {d:?}; there are {}",
                devices.join(", ")
            )))
        }
        Some(d) => vec![d.to_owned()],
        None => devices,
    };
    let mut reasons = Vec::new();
    for name in candidates {
        let ctx = match Context::open(provider.clone(), &name) {
            Ok(ctx) => ctx,
            Err(e) => {
                reasons.push(format!("{name}: {e}"));
                continue;
            }
        };
        let count = ctx.attr().phys_port_cnt;
        let ports: Vec<u8> = match port {
            Some(p) => vec![p],
            None => (1..=count).collect(),
        };
        for p in ports {
            let attr = match ctx.query_port(p) {
                Ok(a) => a,
                Err(e) => {
                    reasons.push(format!("{name} port {p}: {e}"));
                    continue;
                }
            };
            if let Some(why) = port_problem(&attr) {
                reasons.push(format!("{name} port {p}: {why}"));
                continue;
            }
            let chosen = match attr.link_layer {
                LinkLayer::Ethernet => {
                    pick_gid(&ctx, p, &attr, gid_index).map(|(i, g, t)| (Some(i), g, Some(t)))
                }
                LinkLayer::InfiniBand => match gid_index {
                    Some(_) => {
                        pick_gid(&ctx, p, &attr, gid_index).map(|(i, g, t)| (Some(i), g, Some(t)))
                    }
                    None => Ok((
                        None,
                        ctx.query_gid(p, 0).map(|e| e.gid).unwrap_or_default(),
                        None,
                    )),
                },
            };
            match chosen {
                Ok((gid_index, gid, gid_type)) => {
                    return Ok(Selected {
                        ctx,
                        port: p,
                        attr,
                        gid_index,
                        gid,
                        gid_type,
                    })
                }
                Err(why) => reasons.push(format!("{name} port {p}: {why}")),
            }
        }
    }
    Err(Error::Unsupported(format!(
        "no usable RDMA port: {}",
        reasons.join("; ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Header {
        Header {
            kind: Kind::Data,
            seq: 7,
            chunk_len: 100,
            data_freed: u32::MAX,
            ctrl_freed: 3,
            key: 0,
            tag: 0xdead_beef_0000_0001,
            msg_len: 1 << 40,
            offset: (1 << 40) - 100,
        }
    }

    #[test]
    fn headers_round_trip_and_bad_ones_are_refused() {
        let h = header();
        let b = h.encode();
        assert_eq!(Header::decode(&b, 148).unwrap(), h);
        assert!(Header::decode(&b, 147).unwrap_err().contains("147 bytes"));
        let mut bad = b;
        bad[0] ^= 1;
        assert!(Header::decode(&bad, 148).unwrap_err().contains("magic"));
        let mut v1 = b;
        v1[2] = 1;
        assert!(Header::decode(&v1, 148)
            .unwrap_err()
            .contains("protocol version 1"));
        let past_end = Header {
            offset: (1 << 40) - 99,
            ..h
        };
        assert!(Header::decode(&past_end.encode(), 148).is_err());
        let credit = Header {
            kind: Kind::Credit,
            chunk_len: 0,
            ..h
        };
        assert_eq!(
            Header::decode(&credit.encode(), HEADER_BYTES as u32).unwrap(),
            credit
        );
        // A write grant round-trips, and its key never shows in Debug output.
        let grant = Header {
            kind: Kind::Cts,
            chunk_len: 0,
            key: 0x1234_5678,
            msg_len: 4096,
            offset: 0x7f00_0000_1000,
            ..h
        };
        let n = HEADER_BYTES as u32;
        assert_eq!(Header::decode(&grant.encode(), n).unwrap(), grant);
        let shown = format!("{grant:?}");
        assert!(
            !shown.contains("key") && !shown.contains("305419896"),
            "{shown}"
        );
        // A key outside a grant, a grant without an address, an empty rendezvous: refused.
        let stray = Header {
            kind: Kind::Fin,
            ..grant
        };
        assert!(Header::decode(&stray.encode(), n)
            .unwrap_err()
            .contains("remote key"));
        let nowhere = Header { offset: 0, ..grant };
        assert!(Header::decode(&nowhere.encode(), n)
            .unwrap_err()
            .contains("address"));
        let empty = Header {
            kind: Kind::Go,
            key: 0,
            msg_len: 0,
            ..grant
        };
        assert!(Header::decode(&empty.encode(), n)
            .unwrap_err()
            .contains("empty"));
        assert!(Kind::Fin.data_class() && Kind::Cts.data_class() && !Kind::Bye.data_class());
    }

    #[test]
    fn handshakes_round_trip_and_a_newer_version_is_refused() {
        let info = Info {
            version: PROTOCOL_VERSION,
            link_layer: LinkLayer::Ethernet,
            lid: 0,
            gid: Gid([1; 16]),
            grh: true,
            active_mtu: Mtu::Mtu1024,
            slot_bytes: 65536,
            credits: 32,
            ctrl_credits: 4,
            rndv: 1 << 16,
            qpns: vec![0, 0x101],
            psns: vec![0, 0xabcdef],
            device: "roce0".into(),
            port: 1,
        };
        let text = info.encode();
        assert!(!text.contains("rkey"));
        assert_eq!(Info::decode(&text).unwrap(), info);
        // Unknown keys are ignored.
        assert_eq!(Info::decode(&format!("{text};future=1")).unwrap(), info);
        let older = text.replacen("gbverbs 2;", "gbverbs 1;", 1);
        assert!(Info::decode(&older).unwrap_err().contains("protocol 1"));
        assert!(Info::decode("hello").is_err());
    }

    #[test]
    fn roce_gids_are_ranked() {
        let mut v4 = [0u8; 16];
        v4[10] = 0xff;
        v4[11] = 0xff;
        v4[12] = 10;
        let mut ll = [0u8; 16];
        ll[0] = 0xfe;
        ll[1] = 0x80;
        ll[15] = 1;
        let mut global = [0u8; 16];
        global[0] = 0x20;
        global[15] = 1;
        assert_eq!(gid_rank(&Gid(v4), GidType::RoceV2), Some(0));
        assert_eq!(gid_rank(&Gid(global), GidType::RoceV2), Some(1));
        assert_eq!(gid_rank(&Gid(ll), GidType::RoceV2), Some(2));
        assert_eq!(gid_rank(&Gid(ll), GidType::RoceV1), Some(3));
        assert_eq!(gid_rank(&Gid([0; 16]), GidType::RoceV2), None);
        assert_eq!(gid_rank(&Gid(global), GidType::Ib), None);
    }
}
