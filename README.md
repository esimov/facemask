# facemask

**Facemask** is a simple tool to overlay a medical mask over a person's face. It can be used to modify a profile picture for example to raise awareness about the necessity to protect ourself and others by wearing a medical mask, limiting the pandemic risks of covid-19.

## Install
```bash
$ go get -u -v github.com/esimov/facemask && cd $GOPATH/src/github.com/esimov/facemask
$ go install
```

```bash
$ facemask -h
 ____  __    ___  ____  _  _   __   ____  __ _
(  __)/ _\  / __)(  __)( \/ ) / _\ / ___)(  / )
 ) _)/    \( (__  ) _) / \/ \/    \\___ \ )  (
(__) \_/\_/ \___)(____)\_)(_/\_/\_/(____/(__\_)

Face mask generator
    Version: 1.0.1

  -angle float
    	0.0 is 0 radians and 1.0 is 2*pi radians
  -in string
    	Source image
  -iou float
    	Intersection over union (IoU) threshold (default 0.2)
  -max int
    	Maximum size of face (default 1000)
  -min int
    	Minimum size of face (default 20)
  -out string
    	Destination image
  -scale float
    	Scale detection window by percentage (default 1.1)
  -shift float
    	Shift detection window by percentage (default 0.1)
```

## Run it
```bash
$ facemask -in <input> -out <output>
```

![facemask](https://user-images.githubusercontent.com/883386/78664870-8ef8d880-78dd-11ea-8dd1-7bb1ee0ce2eb.png)


## Author

* Endre Simo ([@simo_endre](https://twitter.com/simo_endre))

## License

Copyright © 2020 Endre Simo

This software is distributed under the MIT license. See the [LICENSE](https://github.com/esimov/facemask/blob/master/LICENSE) file for the full license text.
