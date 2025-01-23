import numpy as np
from skimage.segmentation import slic
from skimage.exposure import rescale_intensity

class MyActivationMapGen():
  def compute_pixel_weights(self, img, gen_img):
    weights = np.sum(np.array(img - gen_img)**2, axis=2)
    weights /= np.max(weights)
    return weights

  def confidence_diff(self, prediction_proba, prediction_proba_gen, label):
    return (prediction_proba[label] - prediction_proba_gen[label])**2

  def generate_activation_map(self, img, label, classifier, generator):
    activation_map = np.zeros(img.shape[:2])
    prediction_proba = classifier.predict(np.array([img]), verbose=0)[0]

    for exp in np.arange(1.0, 2.1, 0.05):
      epsilon = -10**(-exp)
      inputs = [np.array([img]), epsilon * np.ones([1, 28, 28, 512])]
      outputs = generator.predict(inputs, verbose=0)
      output_img = outputs[0][0]

      px_weights = self.compute_pixel_weights(img, output_img)
      prediction_proba_gen = classifier.predict(np.array([output_img]), verbose=0)[0]
      activation_map += px_weights * self.confidence_diff(prediction_proba,
                                                    prediction_proba_gen,
                                                    label)
    return activation_map


class SuperPixelActivationMapGen():
  def compute_pixel_weights(self, img, gen_img):
    weights = np.sum(np.array(img - gen_img)**2, axis=2)
    weights /= np.max(weights)
    return weights

  def confidence_diff(self, prediction_proba, prediction_proba_gen, label):
    return (prediction_proba[label] - prediction_proba_gen[label])**2

  def random_toggle_superpixels(self, img, segments, i_segment):
    mask = np.ones(img.shape[:2])
    mask[segments == i_segment] = 0
    mask = np.dstack([mask] * 3)
    return mask * img

  def generate_activation_map(self, img, label, classifier, num_segments=128):
    activation_map = np.zeros(img.shape[:2])
    prediction_proba = classifier.predict(np.array([img]), verbose=0)[0]

    segments = slic(img * 255., n_segments=num_segments, slic_zero=True)
    for i_segment in np.unique(segments):
      output_img = self.random_toggle_superpixels(img, segments, i_segment)

      px_weights = self.compute_pixel_weights(img, output_img)
      prediction_proba_gen = classifier.predict(np.array([output_img]), verbose=0)[0]
      activation_map += px_weights * self.confidence_diff(prediction_proba,
                                                    prediction_proba_gen,
                                                    label)

    return activation_map


class PixelWiseActivationMapGen():
  def compute_pixel_weights(self, img, gen_img):
    weights = np.sum(np.array(img - gen_img)**2, axis=2)
    weights /= np.max(weights)
    return weights

  def confidence_diff(self, prediction_proba, prediction_proba_gen, label):
    return (prediction_proba[label] - prediction_proba_gen[label])**2

  def random_toggle_pixels(self, img):
    h, w, c = img.shape
    mask = np.random.randint(2, size=[h, w])
    mask = np.dstack([mask] * 3)
    return mask * img

  def generate_activation_map(self, img, label, classifier, num_trial=16):
    activation_map = np.zeros(img.shape[:2])
    prediction_proba = classifier.predict(np.array([img]), verbose=0)[0]

    for i in range(num_trial):
      output_img = self.random_toggle_pixels(img)

      px_weights = self.compute_pixel_weights(img, output_img)
      prediction_proba_gen = classifier.predict(np.array([output_img]), verbose=0)[0]
      activation_map += px_weights * self.confidence_diff(prediction_proba,
                                                    prediction_proba_gen,
                                                    label)

    return activation_map