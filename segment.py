import cv
import cv2
import numpy as np
import scipy.ndimage as ndi

class Segment(object):
    def __init__(self):
        self.input_image = None
        self.local_range = None

        # Calculate the local image range footprint
        radius = 3
        self._footprint = np.zeros((2*radius+1, 2*radius+1), dtype=np.bool)
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                d_sq = dx*dx + dy*dy
                if d_sq > radius * radius:
                    continue
                self._footprint[dx + radius, dy + radius] = True

    def segment(self, image):
        self.input_image = image

        # compute the local range image
        self.local_range = ndi.maximum_filter(self.input_image, footprint=self._footprint) 
        self.local_range -= ndi.minimum_filter(self.input_image, footprint=self._footprint) 

        # normalize it
        self.local_range = self.local_range / float(np.amax(self.local_range))

        # find a threshold which gives the coin border
        best_threshold = 0
        best_contour = None
        best_form_factor = 0.0
        best_bin_im = None

        for threshold in np.arange(0.05, 0.65, 0.05):
            # Find contours in thresholded image
            contour_im = self.local_range >= threshold
            contours, _ = cv2.findContours(np.array(contour_im, dtype=np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE)

            # Find maximum area contour
            areas = list(cv2.contourArea(c) for c in contours)
            max_index = np.argmax(areas)

            # Calculate the form factor
            contour = contours[max_index]
            area = areas[max_index]
            perim = cv2.arcLength(contour, closed=True)
            form_factor = 4.0 * np.pi * area / (perim * perim)
            
            # Reject contours with an area > 90% of the image to reject
            # contour covering entire image
            if area > 0.9 * np.product(self.local_range.shape):
                continue
            
            # Update best form factor
            if form_factor >= best_form_factor:
                best_threshold = threshold
                best_contour = contour
                best_form_factor = form_factor
                best_bin_im = contour_im 
        
        # Store the extracted edge
        self.edge = np.reshape(best_contour, (len(best_contour), 2))
        self.edge_mask = best_bin_im
        self.edge_threshold = best_threshold
        self.edge_form_factor = best_form_factor

        # Find centre and radius of best-fit circle
        centre, axes, angle = cv2.fitEllipse(best_contour)
        r = np.mean(axes) * 0.5

        delta_points = np.array(self.edge, copy=True)
        delta_points[:,0] -= centre[0]
        delta_points[:,1] -= centre[1]

        radii = np.sqrt(np.sum(delta_points * delta_points, axis=1))
        thetas = np.arctan2(delta_points[:,1], delta_points[:,0])

        # re-arrange theta and radii in increasing theta order
        theta_indices = np.argsort(thetas)
        thetas = thetas[theta_indices]
        radii = radii[theta_indices]

        angles = np.linspace(-np.pi, np.pi, 256, endpoint=False)
        sampled_radii = np.interp(angles, thetas, radii / r)

        self.centre = centre
        self.radius = r
        self.deviations = np.vstack((angles, sampled_radii)).transpose()


# vim:sw=4:sts=4:et
