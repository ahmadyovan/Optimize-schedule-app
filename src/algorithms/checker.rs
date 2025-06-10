use std::collections::HashSet;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;

use super::models::{OptimizedCourse, ScheduleChecker, TimePreferenceRequest };

#[derive(Serialize)]
pub struct ConflictMessage {
    jadwal_a: u32,
    jadwal_b: u32,
    deskripsi: String,
}

#[derive(Serialize)]
pub struct ConflictResult {
    penalty: u32,
    conflicts: Vec<ConflictMessage>,
}

#[derive(Serialize)]
pub struct PreferenceMessage {
    pub id_jadwal: u32,
    pub id_dosen: u32,
    pub hari: u32,
    pub jam_mulai: u32,
    pub deskripsi: String,
}

#[derive(Serialize)]
pub struct PreferenceResult {
    pub penalty: u32,
    pub violations: Vec<PreferenceMessage>,
}

impl ScheduleChecker {
    pub fn new(time_preferences: Vec<TimePreferenceRequest>) -> Self {
        Self {
            time_preferences: time_preferences
            .into_iter()
            .map(|p| (p.id_dosen, p))
            .collect(),
        }
    }

    pub fn calculate_fitness(&self, schedule: &[OptimizedCourse]) -> f32 {
        let conflict_result = self.detect_conflicts(schedule);
        let preference_result = self.check_preferences(schedule);

        (conflict_result.penalty + preference_result.penalty) as f32
    }

    pub fn schedule_messages(&self, schedule: &[OptimizedCourse]) -> (Vec<ConflictMessage>, Vec<PreferenceMessage>) {
        let conflict_result = self.detect_conflicts(schedule);
        let preference_result = self.check_preferences(schedule);

        (conflict_result.conflicts, preference_result.violations)
    }

    // Detects scheduling conflicts such as overlapping classes or conflicting lecturers
    pub fn detect_conflicts(&self, schedule: &[OptimizedCourse]) -> ConflictResult {
        let mut conflicts = Vec::new();

        for (i, a) in schedule.iter().enumerate() {
            for b in &schedule[i + 1..] {
                if a.hari == b.hari && Self::is_overlap(a, b) && a.id_dosen == b.id_dosen {
                    conflicts.push(ConflictMessage {
                        jadwal_a: a.id_jadwal,
                        jadwal_b: b.id_jadwal,
                        deskripsi: format!(
                            "Konflik dosen yang sama: dosen {} di dua kelas berbeda.",
                            a.id_dosen
                        ),
                    });
                }
            }
        }

        ConflictResult {
            penalty: conflicts.len() as u32 * 100,
            conflicts,
        }
    }


    pub fn check_preferences(&self, schedule: &[OptimizedCourse]) -> PreferenceResult {
        let violations: Vec<PreferenceMessage> = schedule.par_iter()
            .filter_map(|course| {
                let pref = self.time_preferences.get(&course.id_dosen)?;

                let hari_idx = (course.hari - 1) as usize;
                let waktu_ok = if course.jam_mulai < 1080 {
                    match hari_idx {
                        0 => pref.senin_pagi,
                        1 => pref.selasa_pagi,
                        2 => pref.rabu_pagi,
                        3 => pref.kamis_pagi,
                        4 => pref.jumat_pagi,
                        _ => false,
                    }
                } else {
                    match hari_idx {
                        0 => pref.senin_malam,
                        1 => pref.selasa_malam,
                        2 => pref.rabu_malam,
                        3 => pref.kamis_malam,
                        4 => pref.jumat_malam,
                        _ => false,
                    }
                };

                if waktu_ok {
                    None
                } else {
                    let waktu_str = if course.jam_mulai < 1080 { "pagi" } else { "malam" };
                    let hari_str = match course.hari {
                        1 => "Senin",
                        2 => "Selasa",
                        3 => "Rabu",
                        4 => "Kamis",
                        5 => "Jumat",
                        _ => "Hari Tidak Dikenal",
                    };

                    Some(PreferenceMessage {
                        id_jadwal: course.id_jadwal,
                        id_dosen: course.id_dosen,
                        hari: course.hari,
                        jam_mulai: course.jam_mulai,
                        deskripsi: format!(
                            "Dosen {} tidak prefer jadwal {} {}.",
                            course.id_dosen, hari_str, waktu_str
                        ),
                    })
                }
            })
            .collect();

        PreferenceResult {
            penalty: (violations.len() as u32) * 100,
            violations,
        }
    }

    #[inline]
    fn is_overlap(a: &OptimizedCourse, b: &OptimizedCourse) -> bool {
        a.jam_mulai < b.jam_akhir && b.jam_mulai < a.jam_akhir
    }
}
