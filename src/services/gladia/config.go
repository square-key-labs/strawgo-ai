package gladia

type LanguageConfig struct {
	Languages     []string `json:"languages,omitempty"`
	CodeSwitching *bool    `json:"code_switching,omitempty"`
}

type PreProcessingConfig struct {
	AudioEnhancer   *bool    `json:"audio_enhancer,omitempty"`
	SpeechThreshold *float64 `json:"speech_threshold,omitempty"`
}

type CustomVocabularyItem struct {
	Value          string   `json:"value"`
	Intensity      float64  `json:"intensity"`
	Pronunciations []string `json:"pronunciations,omitempty"`
	Language       string   `json:"language,omitempty"`
}

type CustomVocabularyConfig struct {
	Vocabulary       []any    `json:"vocabulary,omitempty"`
	DefaultIntensity *float64 `json:"default_intensity,omitempty"`
}

type CustomSpellingConfig struct {
	SpellingDictionary map[string][]string `json:"spelling_dictionary,omitempty"`
}

type TranslationConfig struct {
	TargetLanguages         []string `json:"target_languages,omitempty"`
	Model                   string   `json:"model,omitempty"`
	MatchOriginalUtterances *bool    `json:"match_original_utterances,omitempty"`
	Lipsync                 *bool    `json:"lipsync,omitempty"`
	ContextAdaptation       *bool    `json:"context_adaptation,omitempty"`
	Context                 string   `json:"context,omitempty"`
	Informal                *bool    `json:"informal,omitempty"`
}

type RealtimeProcessingConfig struct {
	WordsAccurateTimestamps *bool                   `json:"words_accurate_timestamps,omitempty"`
	CustomVocabulary        *bool                   `json:"custom_vocabulary,omitempty"`
	CustomVocabularyConfig  *CustomVocabularyConfig `json:"custom_vocabulary_config,omitempty"`
	CustomSpelling          *bool                   `json:"custom_spelling,omitempty"`
	CustomSpellingConfig    *CustomSpellingConfig   `json:"custom_spelling_config,omitempty"`
	Translation             *bool                   `json:"translation,omitempty"`
	TranslationConfig       *TranslationConfig      `json:"translation_config,omitempty"`
	NamedEntityRecognition  *bool                   `json:"named_entity_recognition,omitempty"`
	SentimentAnalysis       *bool                   `json:"sentiment_analysis,omitempty"`
}

type MessagesConfig struct {
	ReceivePartialTranscripts       *bool `json:"receive_partial_transcripts,omitempty"`
	ReceiveFinalTranscripts         *bool `json:"receive_final_transcripts,omitempty"`
	ReceiveSpeechEvents             *bool `json:"receive_speech_events,omitempty"`
	ReceivePreProcessingEvents      *bool `json:"receive_pre_processing_events,omitempty"`
	ReceiveRealtimeProcessingEvents *bool `json:"receive_realtime_processing_events,omitempty"`
	ReceivePostProcessingEvents     *bool `json:"receive_post_processing_events,omitempty"`
	ReceiveAcknowledgments          *bool `json:"receive_acknowledgments,omitempty"`
	ReceiveErrors                   *bool `json:"receive_errors,omitempty"`
	ReceiveLifecycleEvents          *bool `json:"receive_lifecycle_events,omitempty"`
}
