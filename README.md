# CAPTCHA SOLVER - Solving captchas with Data SCience and Machine Learning with the aid of tensorflow
###### Emil Tu
This project leverages the Tensorflow Object Detection API to automatically solve Google reCAPTCHAs.

The procedure is as follows:

1. Locate and click the reCAPTCHA checkbox.
2. Locate the reCAPTCHA text and image(s).
3. Read the text using OCR.
4. Load the appropriate model based on the text.
5. Denoise the image using Nvidia's Noise2Noise implementation.
6. Classify the reCaptcha based on type (3x3 grid, 4x4 grid, 2x4 grid).
7. Detect objects within the captcha.
8. Click the appropriate boxes and verify.

## Details
The process of detecting objects within the reCAPTCHA relies on Google's Faster-RCNN NASNet architecture, which is a powerful deep learning model. To train this model, a dataset was compiled from the Open Images V4 Dataset, containing images of various objects. The training process utilized a Titan RTX GPU with a batch size of 1. Each object class, including cars, buses, bicycles, fire hydrants, and traffic lights, was trained using a varying number of images ranging from 400 to 17,000, ensuring comprehensive coverage.

For the detection of checkboxes and the reCAPTCHA itself, a different architecture called SSD Mobilenet was chosen. This lightweight model is better suited for this specific task. The training data for these models comprised approximately 100 screenshots that were manually captured.

To classify reCaptchas accurately, a Tensorflow classification model was employed. This model was trained on approximately 300 screenshots, enabling it to effectively differentiate between different types of reCaptchas, such as 3x3, 4x4, and 2x4. The classification results are crucial as they determine the appropriate coordinates for each square of the captcha during the solving process.

To improve the quality of the reCAPTCHA images and enhance the accuracy of the solving algorithm, a denoising algorithm was implemented. This algorithm utilizes Nvidia's noise2noise library and is trained on a dataset of 15,000 reCaptcha images that were collected. Notably, the noise2noise library does not require labeled data, making it easier to leverage large datasets. Denoising is particularly beneficial in reCAPTCHA solving since Google has started incorporating adversarial noise in a subset of images, making it more challenging for traditional algorithms.

Overall, the combination of these techniques and models enables efficient and accurate detection, classification, and solving of reCAPTCHA challenges, providing a robust solution for automating the process while considering the evolving complexity introduced by Google's countermeasures.

## Getting Started
### Hardware
A GPU capable of inference on the NASNet architecture is required (approximately 8GB of VRAM or more).

### Python dependencies
This project was run on Ubuntu 19.04 running Python3.7. Dependencies include pyautogui, pillow, numpy, matplotlib, and tensorflow. By default, it opens the Chrome browser, but this is easy to change according to preference. These can be installed by running the following:
```
pip3 install -r requirements.txt
```
Note: I recommend either compiling Tensorflow yourself, or installing the tensorflow-gpu package for best results.

### Models
The models are hosted in google drive due to their size which GitHub can not host(About 2gb)

|Class |	URL |
|-------------|-------------|
|Bicycle |	[link](https://drive.google.com/file/d/19dSW-_TfIY03s-0xjwmqQrlkjXy0dzcr/view?usp=sharing) |
|Bus | [link](https://drive.google.com/file/d/1fGFZpI3IsVIhW4bKc7_-UQjkHmYg_knv/view?usp=sharing) |
|Captcha image(s) |	[link](https://drive.google.com/file/d/1N0yMl2f5nT1eFTZvK6QpHQ33uexMeayM/view?usp=sharing) |
|Car	| [link](https://drive.google.com/file/d/1qUA0PRJmtNINpS7bdpT0wur19Fd1EKLN/view?usp=sharing) |
|Captcha checkbox	| [link](https://drive.google.com/file/d/11MIzTNSrGRU66Qws-EH0WfXEGVFssCCz/view?usp=sharing) |
|Fire Hydrant	| [link](https://drive.google.com/file/d/1pYbTFR2_XseQ937Yoeih93ediyZnifbu/view?usp=sharing) |
|Traffic Light	| [link](https://drive.google.com/file/d/1GC2LTI2U_nNlX08__HQ97V2QjgO00_Ey/view?usp=sharing) |

Each model is compressed into a tar.gz, and should be extracted into the object_detection directory.

### Usage

Simply run

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python3 run.py
```
