use rayon::prelude::*;
use std::collections::{HashSet};

use super::models::{TimePreferenceRequest, OptimizedCourse, FitnessCalculator};

impl FitnessCalculator {
    pub fn new(time_preferences: Vec<TimePreferenceRequest>) -> Self {
        Self {
            time_preferences: time_preferences.into_iter()
                .map(|p| (p.id_dosen, p))
                .collect(),
        }
    }

    pub fn calculate_fitness(&self, schedule: &[OptimizedCourse]) -> f32 {
        let conflict_penalty = self.detect_conflicts(schedule);
        let preference_penalty = self.check_preferences(schedule);

        (conflict_penalty + preference_penalty) as f32
    }

    pub fn detect_conflicts(&self, schedule: &[OptimizedCourse]) -> u32 {
        let mut penalty = 0;
        let mut checked = HashSet::new();

        for (i, a) in schedule.iter().enumerate() {
            for b in &schedule[i + 1..] {
                if a.hari != b.hari || !Self::is_overlap(a, b) {
                    continue;
                }

                let same_room_class = a.prodi == b.prodi && a.semester == b.semester && a.id_kelas == b.id_kelas;
                let same_dosen = a.id_dosen == b.id_dosen;

                if same_room_class || same_dosen {
                    let key = (a.id_jadwal.min(b.id_jadwal), a.id_jadwal.max(b.id_jadwal));
                    if checked.contains(&key) {
                        continue;
                    }
                    checked.insert(key);
                    penalty += 100;
                }
            }
        }

        penalty
    }

    pub fn check_preferences(&self, schedule: &[OptimizedCourse]) -> u32 {
        schedule.par_iter()
            .filter_map(|course| {
                let pref = self.time_preferences.get(&course.id_dosen)?;

                let hari_idx = (course.hari - 1) as usize;
                let waktu_ok = if course.jam_mulai < 1080 {
                    match hari_idx {
                        0 => pref.seninPagi,
                        1 => pref.selasaPagi,
                        2 => pref.rabuPagi,
                        3 => pref.kamisPagi,
                        4 => pref.jumatPagi,
                        _ => false,
                    }
                } else {
                    match hari_idx {
                        0 => pref.seninMalam,
                        1 => pref.selasaMalam,
                        2 => pref.rabuMalam,
                        3 => pref.kamisMalam,
                        4 => pref.jumatMalam,
                        _ => false,
                    }
                };

                if waktu_ok {
                    None
                } else {
                    Some(100)
                }
            })
            .sum()
    }

    #[inline]
    fn is_overlap(a: &OptimizedCourse, b: &OptimizedCourse) -> bool {
        a.jam_mulai < b.jam_akhir && b.jam_mulai < a.jam_akhir
    }
}
