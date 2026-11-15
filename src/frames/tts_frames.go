package frames

type TTSSpeakFrame struct {
	*DataFrame
	Text            string
	AppendToContext bool
	TextType        TextType
}

func NewTTSSpeakFrame(text string) *TTSSpeakFrame {
	return &TTSSpeakFrame{
		DataFrame:       newDataFrame("TTSSpeakFrame"),
		Text:            text,
		AppendToContext: true,
		TextType:        TextTypeSentence,
	}
}

type BotBackchannelFrame struct {
	*DataFrame
	Text            string
	AppendToContext bool
	TextType        TextType
}

func NewBotBackchannelFrame(text string) *BotBackchannelFrame {
	return &BotBackchannelFrame{
		DataFrame: newUninterruptibleDataFrame("BotBackchannelFrame"),
		Text:      text,
		TextType:  TextTypeBackchannel,
	}
}

type TTSMetadataFrame struct {
	*ServiceMetadataFrame
}

func NewTTSMetadataFrame(serviceName string) *TTSMetadataFrame {
	return &TTSMetadataFrame{ServiceMetadataFrame: newServiceMetadataFrame("TTSMetadataFrame", serviceName)}
}

type STTMuteFrame struct {
	*SystemFrame
	Mute bool
}

func NewSTTMuteFrame(mute bool) *STTMuteFrame {
	return &STTMuteFrame{
		SystemFrame: newSystemFrame("STTMuteFrame"),
		Mute:        mute,
	}
}
