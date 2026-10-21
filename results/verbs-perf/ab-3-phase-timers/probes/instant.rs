// Cost of what the window timers add per window: Instant::now() then elapsed().as_nanos().
use std::time::Instant;
fn main() {
    let n = 5_000_000u64;
    let mut best = f64::MAX;
    let mut sink = 0u64;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..n {
            let s = Instant::now();
            sink = sink.wrapping_add(u64::try_from(s.elapsed().as_nanos()).unwrap_or(u64::MAX));
        }
        best = best.min(t.elapsed().as_nanos() as f64 / n as f64);
    }
    println!("{{\"ns_per_timed_region\": {best:.1}, \"sink\": {}}}", sink % 2);
}
