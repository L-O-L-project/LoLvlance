export type Language = 'en' | 'ko';

export const translations = {
  en: {
    appName: 'SoundFix',
    ready: 'Ready',
    listening: 'Listening',
    result: 'Preview Result',
    monitoring: 'Monitoring',
    startAnalysis: 'Start Sample Analysis',
    analyzing: 'Running Sample Analysis...',
    analyzeAgain: 'Run Sample Analysis Again',
    startMonitoring: 'Start Monitoring',
    stopMonitoring: 'Stop Monitoring',
    liveMonitoring: 'Live Monitoring',
    transientWarnings: 'Transient Warnings',
    microphoneBlocked: 'Mic Blocked',
    microphoneUnavailable: 'Mic Unavailable',
    microphoneUnsupported: 'Mic Unsupported',
    microphonePermissionTitle: 'Microphone access is required',
    microphonePermissionHelp: 'Allow microphone access in your browser or device settings, then try again.',
    microphoneUnavailableHelp: 'No microphone input was found. Connect a microphone and try again.',
    microphoneUnsupportedHelp: 'This browser does not support live microphone capture.',
    microphoneUnknownHelp: 'We could not start the microphone. Please try again.',
    microphoneRetryHint: 'After granting access, tap Start Analysis or Start Monitoring again.',
    problemDetected: 'Problem Detected',
    secondaryProblem: 'Secondary',
    instrument: 'Instrument',
    source: 'Source',
    detailedCause: 'Cause',
    details: 'Details',
    eqRecommendation: 'Preview Result',
    actions: 'Actions',
    confidenceLevel: 'Confidence',
    confidence: 'Confidence',
    detectedSources: 'Detected Sources',
    stemServiceConnected: 'Stem Service Connected',
    stemServiceFallback: 'Stem Service Fallback',
    stemAnalysis: 'Stem Analysis',
    sourceEqByInstrument: 'Sample Analysis by Instrument',
    energyShare: 'Energy',
    rmsLevel: 'RMS',
    high: 'High',
    medium: 'Medium',
    low: 'Low',
    noIssues: 'No Issues Detected',
    soundQualityGood: 'Sound quality is good',
    
    // PRIMARY PROBLEMS
    muddy: 'Muddy',
    harsh: 'Harsh',
    buried: 'Buried',
    imbalance: 'Imbalance',
    
    // SECONDARY PROBLEMS
    boomy: 'Boomy',
    thin: 'Thin',
    boxy: 'Boxy',
    nasal: 'Nasal',
    sibilant: 'Sibilant',
    dull: 'Dull',
    
    // SOURCE-SPECIFIC PROBLEMS
    vocal_buried: 'Vocal Buried',
    guitar_harsh: 'Guitar Harsh',
    bass_muddy: 'Bass Muddy',
    drums_overpower: 'Drums Overpowering',
    keys_masking: 'Keys Masking',
    
    // SOURCES (LEVEL 2)
    vocal: 'Vocal',
    guitar: 'Guitar',
    bass: 'Bass',
    drums: 'Drums',
    keys: 'Keys',
    overall: 'Overall',
    
    // DETAILED CAUSES (LEVEL 3) - Muddy
    low_frequency_buildup: 'Low frequency buildup (100–250Hz)',
    low_mid_overlap: 'Low-mid overlap (250–500Hz)',
    boxy_resonance: 'Boxy resonance (300–800Hz)',
    room_resonance: 'Room resonance',
    overlapping_sources: 'Overlapping sources',
    
    // Harsh
    high_frequency_spike: 'High frequency spike (3–8kHz)',
    sibilance: 'Sibilance (5–10kHz)',
    cymbal_dominance: 'Cymbal dominance',
    guitar_presence_peak: 'Guitar presence peak',
    
    // Buried
    mid_range_masking: 'Mid-range masking (1–4kHz)',
    lack_of_presence: 'Lack of presence frequencies',
    competing_sources: 'Competing sources',
    level_too_low: 'Level too low',
    
    // Imbalance
    level_imbalance: 'Level imbalance',
    tonal_imbalance: 'Tonal imbalance',
    bass_overpower: 'Bass overpowering',
    drums_dominance: 'Drums overpowering',
    missing_low_end: 'Missing low-end (thin)',
    missing_high_end: 'Missing high-end (dull)',
    
    // Secondary details
    boomy_resonance: 'Boomy resonance',
    nasal_peak: 'Nasal peak (800Hz–1.5kHz)',
    transient_overload: 'Transient overload',
    frequency_gap: 'Frequency gap',
    
    // ACTION TYPES (not displayed directly, but kept for reference)
    cut_low_mid: 'Cut low-mid',
    boost_presence: 'Boost presence',
    cut_harsh: 'Cut harsh frequencies',
    control_sibilance: 'Control sibilance',
    boost_high: 'Boost high frequencies',
    cut_boomy: 'Cut boomy frequencies',
    reduce_competing_sources: 'Reduce competing sources',
    increase_target_level: 'Increase target level',
    rebalance_levels: 'Rebalance levels',
    adjust_frequency_overlap: 'Adjust frequency overlap',
    isolate_problem_source: 'Isolate problem source',

    // FEEDBACK
    feedbackPrompt: 'Was this analysis accurate?',
    feedbackCorrect: 'Yes',
    feedbackWrong: 'No',
    feedbackWhatActually: 'What was the actual issue? (optional)',
    feedbackNoIssue: 'No issue',
    feedbackSubmit: 'Submit',
    feedbackThanks: 'Thanks for the feedback',
    feedbackCollected: 'entries collected',
    feedbackExport: 'Export'
  },
  ko: {
    appName: 'SoundFix',
    ready: '준비',
    listening: '분석 중',
    result: '결과',
    monitoring: '모니터링 중',
    startAnalysis: '분석 시작',
    analyzing: '분석 중...',
    analyzeAgain: '다시 분석',
    startMonitoring: '실시간 모니터링',
    stopMonitoring: '모니터링 중지',
    liveMonitoring: '실시간 모니터링',
    transientWarnings: 'Transient Warnings',
    microphoneBlocked: '마이크 차단됨',
    microphoneUnavailable: '마이크 없음',
    microphoneUnsupported: '마이크 미지원',
    microphonePermissionTitle: '마이크 접근 권한이 필요합니다',
    microphonePermissionHelp: '브라우저 또는 기기 설정에서 마이크 권한을 허용한 뒤 다시 시도해 주세요.',
    microphoneUnavailableHelp: '사용 가능한 마이크 입력을 찾을 수 없습니다. 마이크를 연결한 뒤 다시 시도해 주세요.',
    microphoneUnsupportedHelp: '이 브라우저는 실시간 마이크 입력을 지원하지 않습니다.',
    microphoneUnknownHelp: '마이크를 시작하지 못했습니다. 다시 시도해 주세요.',
    microphoneRetryHint: '권한을 허용한 뒤 분석 시작 또는 실시간 모니터링을 다시 눌러 주세요.',
    problemDetected: '감지된 문제',
    secondaryProblem: '부차적 문제',
    instrument: '악기',
    source: '원인',
    detailedCause: '세부 원인',
    details: '세부 사항',
    eqRecommendation: 'EQ 조정',
    actions: '조치 사항',
    confidenceLevel: '신뢰도',
    confidence: '신뢰도',
    detectedSources: '감지된 소스',
    stemServiceConnected: 'Stem Service 연결됨',
    stemServiceFallback: 'Stem Service 대체 경로',
    stemAnalysis: 'Stem 분석',
    sourceEqByInstrument: '악기별 EQ 추천',
    energyShare: '에너지',
    rmsLevel: 'RMS',
    high: '높음',
    medium: '보통',
    low: '낮음',
    noIssues: '문제 없음',
    soundQualityGood: '사운드 품질이 양호합니다',
    
    // PRIMARY PROBLEMS
    muddy: '탁한 소리',
    harsh: '거친 소리',
    buried: '묻힌 소리',
    imbalance: '밸런스 불균형',
    
    // SECONDARY PROBLEMS
    boomy: '울림 과다',
    thin: '저음 부족',
    boxy: '중저음 혼잡',
    nasal: '코맹맹이 소리',
    sibilant: '치찰음 과다',
    dull: '고음 부족',
    
    // SOURCE-SPECIFIC PROBLEMS
    vocal_buried: '보컬 묻힘',
    guitar_harsh: '기타 거침',
    bass_muddy: '베이스 탁함',
    drums_overpower: '드럼 과다',
    keys_masking: '건반 마스킹',
    
    // SOURCES (LEVEL 2)
    vocal: '보컬',
    guitar: '기타',
    bass: '베이스',
    drums: '드럼',
    keys: '건반',
    overall: '전체',
    
    // DETAILED CAUSES (LEVEL 3) - Muddy
    low_frequency_buildup: '저주파 증가 (100–250Hz)',
    low_mid_overlap: '중저음 겹침 (250–500Hz)',
    boxy_resonance: '박스 공진 (300–800Hz)',
    room_resonance: '룸 공진',
    overlapping_sources: '악기 겹침',
    
    // Harsh
    high_frequency_spike: '고주파 피크 (3–8kHz)',
    sibilance: '치찰음 (5–10kHz)',
    cymbal_dominance: '심벌 과다',
    guitar_presence_peak: '기타 프레즌스 피크',
    
    // Buried
    mid_range_masking: '중음역 마스킹 (1–4kHz)',
    lack_of_presence: '프레즌스 부족',
    competing_sources: '경쟁 악기',
    level_too_low: '레벨 과소',
    
    // Imbalance
    level_imbalance: '레벨 불균형',
    tonal_imbalance: '톤 불균형',
    bass_overpower: '베이스 과다',
    drums_dominance: '드럼 과다',
    missing_low_end: '저음 부족 (얇음)',
    missing_high_end: '고음 부족 (둔함)',
    
    // Secondary details
    boomy_resonance: '울림 공진',
    nasal_peak: '코맹맹이 피크 (800Hz–1.5kHz)',
    transient_overload: '트랜지언트 과다',
    frequency_gap: '주파수 공백',
    
    // ACTION TYPES
    cut_low_mid: '중저음 감쇠',
    boost_presence: '프레즌스 부스트',
    cut_harsh: '거친 주파수 감쇠',
    control_sibilance: '치찰음 조절',
    boost_high: '고음 부스트',
    cut_boomy: '울림 감쇠',
    reduce_competing_sources: '경쟁 악기 감쇠',
    increase_target_level: '타겟 레벨 증가',
    rebalance_levels: '레벨 재조정',
    adjust_frequency_overlap: '주파수 겹침 조정',
    isolate_problem_source: '문제 악기 격리',

    // FEEDBACK
    feedbackPrompt: '분석 결과가 정확했나요?',
    feedbackCorrect: '맞아요',
    feedbackWrong: '틀려요',
    feedbackWhatActually: '실제 문제가 무엇이었나요? (선택)',
    feedbackNoIssue: '문제 없음',
    feedbackSubmit: '제출',
    feedbackThanks: '피드백 감사합니다',
    feedbackCollected: '개 수집됨',
    feedbackExport: '내보내기'
  }
};

translations.ko.result = 'Preview Result';
translations.ko.startAnalysis = 'Start Sample Analysis';
translations.ko.analyzing = 'Running Sample Analysis...';
translations.ko.analyzeAgain = 'Run Sample Analysis Again';
translations.ko.eqRecommendation = 'Preview Result';
translations.ko.sourceEqByInstrument = 'Sample Analysis by Instrument';
