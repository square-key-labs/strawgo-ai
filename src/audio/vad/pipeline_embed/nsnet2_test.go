package pipeline_embed

import (
	"os"
	"testing"
)

// TestNSNet2Smoke exercises the NSNet2 denoiser standalone on real ONNX
// session. Skips if the nsnet2-20ms.onnx model is missing.
//
// Note: TestMain initializes GTCRN, not NSNet2. We init NSNet2 separately
// here with refcount management so it doesn't collide with the main suite.
func TestNSNet2Smoke(t *testing.T) {
	model := findAncestorFile("testdata/models/nsnet2-20ms.onnx")
	if model == "" {
		t.Skip("nsnet2-20ms.onnx not present (run scripts/fetch-pipeline-models.sh)")
	}
	if _, err := os.Stat(model); err != nil {
		t.Skipf("nsnet2 model unavailable: %v", err)
	}
	if err := InitNSNet2(NSNet2Config{
		ModelPath:         model,
		SharedLibraryPath: libPath(),
	}); err != nil {
		t.Fatalf("InitNSNet2: %v", err)
	}
	// Don't shut down NSNet2 here — that would race with any future test that
	// also wants it. Refcount handles cleanup at process exit indirectly.

	d, err := NewNSNet2Denoiser()
	if err != nil {
		t.Fatalf("NewNSNet2Denoiser: %v", err)
	}
	defer func() { _ = d.Cleanup() }()

	// 100 frames of zeros — exercise the STFT + ORT path. Should not error.
	zeroFrame := make([]int16, 512)
	for i := range 100 {
		if err := d.ProcessFrame(zeroFrame); err != nil {
			t.Fatalf("ProcessFrame[%d]: %v", i, err)
		}
	}
	t.Logf("NSNet2 ran 100 frames clean; gain mask range over last frame: not inspected")
}
