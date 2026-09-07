export interface SkillProgress {
  task_type: string
  label: string
  family: string
  level: string
  status: "locked" | "practice" | "mastered" | "needs_sft"
  accuracy: number | null
  samples: number
  checks: number
  invalid_answers: number
  truncated: number
}

export interface MasteryEvent {
  task_type: string
  step: number
  action: string
  correct: number
  samples: number
}

export interface PracticeTest {
  accuracy: number
  samples_per_task: number
  tasks: {
    task_type: string
    accuracy: number
    invalid_answers: number
    truncated: number
    samples: number
  }[]
}

export interface PracticeReport {
  mastery?: SkillProgress[]
  curriculum_events?: MasteryEvent[]
  test_before?: PracticeTest
  test_after?: PracticeTest
  practice_phase?: string
}
