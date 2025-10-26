export const SCHOOLS = [
  { id: 1, name: "National University of Singapore (NUS)" },
  { id: 2, name: "Nanyang Technological University (NTU)" },
  { id: 3, name: "Singapore Management University (SMU)" },
  { id: 4, name: "Singapore University of Technology and Design (SUTD)" },
  { id: 5, name: "Singapore Institute of Technology (SIT)" },
  { id: 6, name: "Singapore University of Social Sciences (SUSS)" },
] as const

export type School = (typeof SCHOOLS)[number]
