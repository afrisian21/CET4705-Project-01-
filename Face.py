import sys
import cv2
import select
import config
import time
import RPi.GPIO as GPIO
import face
import hardware
import glob
import os

NONRECON_FILE_PREFIX = 'Intruder_'

def is_letter_input(letter):
	if select.select([sys.stdin,],[],[],0.0)[0]:
		input_char = sys.stdin.read(1)
		return input_char.lower() == letter.lower()
	return False

if __name__ == '__main__':
	print 'Loadign training data...'
	model = cv2.createEigenFaceRecognizer()
	model.load(config.TRAINING_FILE)
	print 'Training data loaded!'

	box = config.get_camera()
	camera = config.get_camera()

	if not os.path.exists(config.NONRECON_DIR):
		os.makedirs(config.NONRECON_DIR)
	files = sorted(glob.glob(os.path.join(config.NONRECON_DIR,
		NONRECON_FILE_PREFIX + '[0-9][0-9][0-9].jpg')))
	count = 0
	if len(files) > 0:
		count = int(files[-1][-7:-4])+1

	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(13, GPIO.OUT)
	GPIO.setup(19, GPIO.OUT)
	GPIO.setup(26, GPIO.OUT)

	print 'Running Recon...'
	print 'press C to turn on Yellow LED (Unlock) and press enter'
	print 'Press Ctl-c to quit.'

	while True:
		if is_letter_input('c'):
			print 'input received, looking for face...'
				
			imgur = camera.read()
			image = camera.read()
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

			result = face.detect_single(image)
			if result is None:
				GPIO.output(26, True)
				print 'Could not detect single face! Check the image in capture.pgm' \
					  ' to see what was captured and try again with only one face visible.'
				time.sleep(1)
				GPIO.output(26, False)
				continue
			x, y, w, h = result
			crop = face.resize(face.crop(image, x, y, w, h))
			label, confidence = model.predict(crop)
			print 'Predicted {0} face with confidence [1] (lower is more confident).'.format(
				'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE',
				confidence)
			if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
				print 'Recognized face!'
				GPIO.output(19, True)
				GPIO.output(13, True)
				time.sleep(1)
				GPIO.output(19, False)
				print 'Yellow LED will turn off after 15s'
				time.sleep(15)
				GPIO.output(13, False)
				print 'To try again press c and then enter'\
					' press ctrl+c to exit'
			else:
				GPIO.output(26, True)
				print 'Did not recognize face!'
				filename = os.path.join(config.NONRECON_DIR, NONRECON_FILE_PREFIX + '%03d.jpg' % count)
				cv2.imwrite(filename, imgur)
				time.sleep(2)
				GPIO.output(26, False)
