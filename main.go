package main

import (
	"flag"
	"fmt"
	"image/color"
	"image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/disintegration/imaging"
	pigo "github.com/esimov/pigo/core"
	"github.com/fogleman/gg"
)

const banner = `
┌─┐┬┌─┐┌─┐
├─┘││ ┬│ │
┴  ┴└─┘└─┘

Go (Golang) Face detection library.
    Version: %s

`

// Version indicates the current build version.
var Version string

var (
	dc        *gg.Context
	fd        *faceDetector
	plc       *pigo.PuplocCascade
	flpcs     map[string][]*pigo.FlpCascade
	imgParams *pigo.ImageParams
)

type point struct {
	x, y int
}

// faceDetector struct contains Pigo face detector general settings.
type faceDetector struct {
	angle        float64
	destination  string
	minSize      int
	maxSize      int
	shiftFactor  float64
	scaleFactor  float64
	iouThreshold float64
	faceCascade  string
	eyesCascade  string
	flplocDir    string
}

func main() {
	var (
		// Flags
		source       = flag.String("in", "", "Source image")
		destination  = flag.String("out", "", "Destination image")
		minSize      = flag.Int("min", 20, "Minimum size of face")
		maxSize      = flag.Int("max", 1000, "Maximum size of face")
		shiftFactor  = flag.Float64("shift", 0.1, "Shift detection window by percentage")
		scaleFactor  = flag.Float64("scale", 1.1, "Scale detection window by percentage")
		angle        = flag.Float64("angle", 0.0, "0.0 is 0 radians and 1.0 is 2*pi radians")
		iouThreshold = flag.Float64("iou", 0.2, "Intersection over union (IoU) threshold")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, fmt.Sprintf(banner, Version))
		flag.PrintDefaults()
	}
	flag.Parse()

	if len(*source) == 0 || len(*destination) == 0 {
		log.Fatal("Usage: pigo -in input.jpg -out out.png -cf cascade/facefinder")
	}

	fileTypes := []string{".jpg", ".jpeg", ".png"}
	ext := filepath.Ext(*destination)

	if !inSlice(ext, fileTypes) {
		log.Fatalf("Output file type not supported: %v", ext)
	}

	if *scaleFactor < 1.05 {
		log.Fatal("Scale factor must be greater than 1.05")
	}

	// Progress indicator
	s := new(spinner)
	s.start("Processing...")
	start := time.Now()

	fd = &faceDetector{
		angle:        *angle,
		destination:  *destination,
		minSize:      *minSize,
		maxSize:      *maxSize,
		shiftFactor:  *shiftFactor,
		scaleFactor:  *scaleFactor,
		iouThreshold: *iouThreshold,
		faceCascade:  "cascades/facefinder",
		eyesCascade:  "cascades/puploc",
		flplocDir:    "cascades/lps",
	}
	faces, err := fd.detectFaces(*source)
	if err != nil {
		log.Fatalf("Detection error: %v", err)
	}

	if err = fd.drawFaces(faces); err != nil {
		log.Fatalf("Error creating the image output: %s", err)
	}

	s.stop()
	fmt.Printf("\nDone in: \x1b[92m%.2fs\n", time.Since(start).Seconds())
}

// detectFaces run the detection algorithm over the provided source image.
func (fd *faceDetector) detectFaces(source string) ([]pigo.Detection, error) {
	src, err := pigo.GetImage(source)
	if err != nil {
		return nil, err
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	dc = gg.NewContext(cols, rows)
	dc.DrawImage(src, 0, 0)

	imgParams = &pigo.ImageParams{
		Pixels: pixels,
		Rows:   rows,
		Cols:   cols,
		Dim:    cols,
	}

	cParams := pigo.CascadeParams{
		MinSize:     fd.minSize,
		MaxSize:     fd.maxSize,
		ShiftFactor: fd.shiftFactor,
		ScaleFactor: fd.scaleFactor,
		ImageParams: *imgParams,
	}

	faceCascade, err := ioutil.ReadFile(fd.faceCascade)
	if err != nil {
		return nil, err
	}

	p := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := p.Unpack(faceCascade)
	if err != nil {
		return nil, err
	}

	pl := pigo.NewPuplocCascade()
	eyesCascade, err := ioutil.ReadFile(fd.eyesCascade)
	if err != nil {
		return nil, err
	}
	plc, err = pl.UnpackCascade(eyesCascade)
	if err != nil {
		return nil, err
	}

	flpcs, err = pl.ReadCascadeDir(fd.flplocDir)
	if err != nil {
		return nil, err
	}

	// Run the classifier over the obtained leaf nodes and return the detection results.
	// The result contains quadruplets representing the row, column, scale and detection score.
	faces := classifier.RunCascade(cParams, fd.angle)

	// Calculate the intersection over union (IoU) of two clusters.
	faces = classifier.ClusterDetections(faces, fd.iouThreshold)

	return faces, nil
}

// drawFaces marks the detected faces with a circle in case isCircle is true, otherwise marks with a rectangle.
func (fd *faceDetector) drawFaces(faces []pigo.Detection) error {
	var (
		qThresh  = float32(5.0)
		perturb  = 63
		puploc   *pigo.Puploc
		imgScale float64
		p1, p2   point
	)

	for _, face := range faces {
		if face.Q > qThresh {
			dc.DrawRectangle(
				float64(face.Col-face.Scale/2),
				float64(face.Row-face.Scale/2),
				float64(face.Scale),
				float64(face.Scale),
			)
			dc.SetLineWidth(2.0)
			dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 255, G: 0, B: 0, A: 255}))
			dc.Stroke()

			// left eye
			puploc = &pigo.Puploc{
				Row:      face.Row - int(0.075*float32(face.Scale)),
				Col:      face.Col - int(0.175*float32(face.Scale)),
				Scale:    float32(face.Scale) * 0.25,
				Perturbs: perturb,
			}
			leftEye := plc.RunDetector(*puploc, *imgParams, fd.angle, false)

			// right eye
			puploc = &pigo.Puploc{
				Row:      face.Row - int(0.075*float32(face.Scale)),
				Col:      face.Col + int(0.185*float32(face.Scale)),
				Scale:    float32(face.Scale) * 0.25,
				Perturbs: perturb,
			}

			rightEye := plc.RunDetector(*puploc, *imgParams, fd.angle, false)

			flp := flpcs["lp84"][0].FindLandmarkPoints(leftEye, rightEye, *imgParams, perturb, false)
			if flp.Row > 0 && flp.Col > 0 {
				drawDetections(dc,
					float64(flp.Col),
					float64(flp.Row),
					float64(flp.Scale*0.5),
					color.RGBA{R: 0, G: 0, B: 255, A: 255},
					false,
				)
			}
			p1 = point{x: flp.Row, y: flp.Col}

			flp = flpcs["lp84"][0].FindLandmarkPoints(leftEye, rightEye, *imgParams, perturb, true)
			if flp.Row > 0 && flp.Col > 0 {
				drawDetections(dc,
					float64(flp.Col),
					float64(flp.Row),
					float64(flp.Scale*0.5),
					color.RGBA{R: 0, G: 0, B: 255, A: 255},
					false,
				)
			}
			p2 = point{x: flp.Row, y: flp.Col}

			mask, err := os.OpenFile("assets/facemask.png", os.O_RDONLY, 0755)
			defer mask.Close()

			if err != nil {
				return err
			}
			maskImg, err := png.Decode(mask)
			if err != nil {
				log.Fatal(err)
			}

			// Calculate the lean angle between the two mouth points.
			angle := 1 - (math.Atan2(float64(p2.y-p1.y), float64(p2.x-p1.x)) * 180 / math.Pi / 90)
			dx, dy := maskImg.Bounds().Dx(), maskImg.Bounds().Dy()

			fmt.Println(face.Scale)
			fmt.Println(dx, dy)
			if face.Scale < dx || face.Scale < dy {
				if dx > dy {
					imgScale = float64(face.Scale) / float64(dx)
				} else {
					imgScale = float64(face.Scale) / float64(dy)
				}
			}
			fmt.Println(imgScale)
			width, height := float64(dx)*imgScale*0.75, float64(dy)*imgScale*0.75
			tx := face.Row - int(width/2*0.8)
			ty := p1.x + (p1.x-p2.x)/2 - int(height/2)

			resized := imaging.Resize(maskImg, int(width), int(height), imaging.Lanczos)
			aligned := imaging.Rotate(resized, angle, color.Transparent)

			fmt.Println(tx, ty)
			fmt.Println(width, height)
			dc.DrawImage(aligned, tx, ty)
		}
	}

	img := dc.Image()
	output, err := os.OpenFile(fd.destination, os.O_CREATE|os.O_RDWR, 0755)
	defer output.Close()

	if err != nil {
		return err
	}
	ext := filepath.Ext(output.Name())

	switch ext {
	case ".jpg", ".jpeg":
		if err := jpeg.Encode(output, img, &jpeg.Options{Quality: 100}); err != nil {
			return err
		}
	case ".png":
		if err := png.Encode(output, img); err != nil {
			return err
		}
	}
	return nil
}

type spinner struct {
	stopChan chan struct{}
}

// Start process
func (s *spinner) start(message string) {
	s.stopChan = make(chan struct{}, 1)

	go func() {
		for {
			for _, r := range `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` {
				select {
				case <-s.stopChan:
					return
				default:
					fmt.Printf("\r%s%s %c%s", message, "\x1b[35m", r, "\x1b[39m")
					time.Sleep(time.Millisecond * 100)
				}
			}
		}
	}()
}

// End process
func (s *spinner) stop() {
	s.stopChan <- struct{}{}
}

// inSlice checks if the item exists in the slice.
func inSlice(item string, slice []string) bool {
	for _, it := range slice {
		if it == item {
			return true
		}
	}
	return false
}

// drawDetections helper function to draw the detection marks
func drawDetections(ctx *gg.Context, x, y, r float64, c color.RGBA, markDet bool) {
	ctx.DrawArc(x, y, r*0.15, 0, 2*math.Pi)
	ctx.SetFillStyle(gg.NewSolidPattern(c))
	ctx.Fill()

	if markDet {
		ctx.DrawRectangle(x-(r*1.5), y-(r*1.5), r*3, r*3)
		ctx.SetLineWidth(2.0)
		ctx.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 255, G: 255, B: 0, A: 255}))
		ctx.Stroke()
	}
}
