package resample

import (
	"math"
	"sync"
)

const (
	baseTaps     = 64
	kaiserBeta   = 8.0
	cutoffFactor = 0.46
	maxPhases    = 512
	interpPhases = 256
)

type bank struct {
	l, m        int
	taps        int
	phases      int
	exact       bool
	coefs       []float32
	delayOutput int
}

type bankKey struct{ l, m int }

var (
	bankMu    sync.Mutex
	bankCache = map[bankKey]*bank{}
)

var commonRates = [][2]int{
	{8000, 16000}, {16000, 8000},
	{8000, 24000}, {24000, 8000},
	{16000, 24000}, {24000, 16000},
	{16000, 48000}, {48000, 16000},
	{24000, 48000}, {48000, 24000},
	{44100, 48000}, {48000, 44100},
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func getBank(inRate, outRate int) *bank {
	g := gcd(inRate, outRate)
	key := bankKey{l: outRate / g, m: inRate / g}
	bankMu.Lock()
	defer bankMu.Unlock()
	if b, ok := bankCache[key]; ok {
		return b
	}
	b := designBank(key.l, key.m)
	bankCache[key] = b
	return b
}

func init() {
	for _, r := range commonRates {
		getBank(r[0], r[1])
	}
}

func designBank(l, m int) *bank {
	taps := baseTaps
	if m > l {
		taps = int(math.Ceil(float64(baseTaps) * float64(m) / float64(l)))
	}
	phases := l
	exact := true
	if l > maxPhases {
		phases = interpPhases
		exact = false
	}
	rows := phases
	if !exact {
		rows = phases + 1
	}
	fc := cutoffFactor / math.Max(float64(l), float64(m))
	if !exact {
		fc = cutoffFactor * math.Min(1, float64(l)/float64(m)) / float64(phases)
	}
	n := phases * taps
	outStep := float64(phases) * float64(m) / float64(l)
	delay := int(math.Round(float64(n-1) / 2 / outStep))
	center := float64(delay) * outStep
	i0b := besselI0(kaiserBeta)
	coefs := make([]float32, rows*taps)
	gain := float64(phases)
	for p := 0; p < rows; p++ {
		for k := 0; k < taps; k++ {
			j := float64(p + phases*k)
			x := j - center
			s := 2 * fc
			if x != 0 {
				s = math.Sin(2*math.Pi*fc*x) / (math.Pi * x)
			}
			r := x / (float64(n) / 2)
			w := 0.0
			if r*r < 1 {
				w = besselI0(kaiserBeta*math.Sqrt(1-r*r)) / i0b
			}
			coefs[p*taps+(taps-1-k)] = float32(gain * s * w)
		}
	}
	normalize(coefs, rows, taps)
	return &bank{l: l, m: m, taps: taps, phases: phases, exact: exact, coefs: coefs, delayOutput: delay}
}

func normalize(coefs []float32, rows, taps int) {
	for p := 0; p < rows; p++ {
		row := coefs[p*taps : (p+1)*taps]
		var sum float64
		for _, c := range row {
			sum += float64(c)
		}
		if sum == 0 {
			continue
		}
		inv := float32(1 / sum)
		for i := range row {
			row[i] *= inv
		}
	}
}

func besselI0(x float64) float64 {
	sum, term := 1.0, 1.0
	half := x / 2
	for k := 1; k < 64; k++ {
		term *= (half / float64(k)) * (half / float64(k))
		sum += term
		if term < sum*1e-12 {
			break
		}
	}
	return sum
}
