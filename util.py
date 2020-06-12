import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

quad_coords = {
    "lonlat": np.array([
        [0, 0], # bottom left
        [7, 0], # bottom right
        [0, 50 ], # top left
        [7, 50] # top right
    ]),
    "pixel": np.array([
        [500, 830], # bottom left
        [1168, 875], # bottom right
        [1280, 616], # top left
        [1350, 608] # top right
    ])
}

def load_images(img_names, model_size):
	"""Loads images in a 4D array.
	Args:
		img_names: A list of images names.
		model_size: The input size of the model.
		data_format: A format for the array returned
		('channels_first' or 'channels_last').
	Returns:
		A 4D NumPy array.
	"""
	imgs = []
	for img_name in img_names:
		img = Image.open(img_name)
		img = img.resize(size=model_size)
		img = np.array(img, dtype=np.float32)
		img = np.expand_dims(img, axis=0)
		imgs.append(img)
		
	imgs = np.concatenate(imgs)
	return imgs
	
def load_class_names(file_name):
	"""Returns a list of class names read from `file_name`."""
	with open(file_name, 'r') as f:
		class_names = f.read().splitlines()
	return class_names


def draw_boxes(img, boxes_dicts, class_names, model_size):
	"""Draws detected boxes.
	Args:
		img_names: A list of input images names.
		boxes_dict: A class-to-boxes dictionary.
		class_names: A class names list.
		model_size: The input size of the model.
	Returns:
		None.
	"""
	for boxes_dict in boxes_dicts:
		img = Image.fromarray(img)
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype(font='data/futur.ttf',
						size=(img.size[0] + img.size[1]) // 100)
		resize_factor = \
				(img.size[0] / model_size[0], img.size[1] / model_size[1])
		#resize_factor = (1280/416,720/416)
		for cls in range(len(class_names)):
			boxes = boxes_dict[cls]
			if np.size(boxes) != 0 and cls==2 :
				for box in boxes:
					xy, confidence = box[:4], box[4]
					xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
					x0, y0 = xy[0], xy[1]
					thickness = 2
					for t in np.linspace(0, 1, thickness):
						xy[0], xy[1] = xy[0] + t, xy[1] + t
						xy[2], xy[3] = xy[2] - t, xy[3] - t
						draw.rectangle(xy, outline=(50,100,150))
					"""text = '{} {:.1f}%'.format(class_names[cls],
										confidence * 100)
					text_size = draw.textsize(text, font=font)
					draw.rectangle(
						[x0, y0 - text_size[1], x0 + text_size[0], y0],
						fill=tuple(color))
					draw.text((x0, y0 - text_size[1]), text, fill='black',
						font=font)"""
					
	return np.array(img)
	
def track_img(img, boxes_dict, model_size, pts):
	"""Draws detected boxes.
	Args:
		img_names: A list of input images names.
		boxes_dict: A dimensions of box.
		model_size: The input size of the model.
	Returns:
		None.
	"""
	mapper = PixelMapper()

	img = Image.fromarray(img)
	draw = ImageDraw.Draw(img)
	#draw.rectangle([(1142,707),(1148, 713)], outline=(50,100,150))
	
	font = ImageFont.truetype(font='data/futur.ttf',
					size=(img.size[0] + img.size[1]) // 100)
	resize_factor = \
			(img.size[0] / model_size[0], img.size[1] / model_size[1])
	#resize_factor = (1280/416,720/416)
	cls = 2
	dist_from_reference = [20]
	boxes = boxes_dict
	if np.size(boxes) != 0 and cls==2 :
		for pt in pts:
			center_points = scaling(pt, resize_factor)
			draw.line(center_points, fill='orange',width=2)
			if len(center_points)>1:
				if (center_points[-1][1]>620):
					distance = np.linalg.norm(mapper.pixel_to_lonlat
						(center_points[-1])- mapper.pixel_to_lonlat((830, 850)))
					speed = int(distance)
					dist_from_reference.append(speed)
					#print(dist_from_reference)
					draw.text(center_points[-1], str(speed), fill = 'white', font=font)
					
		
		for box in boxes:
			xy = box[:4]
			xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
			x0, y0 = xy[0], xy[1]
			thickness = 3
			"""draw.points(pt)   """
			for t in np.linspace(0, 1, thickness):
				xy[0], xy[1] = xy[0] + t, xy[1] + t
				xy[2], xy[3] = xy[2] - t, xy[3] - t
				draw.rectangle(xy, outline=(50,100,150))
			text = '{}'.format(int(box[4]))
			text_size = draw.textsize(text, font=font)
			"""draw.rectangle(
				[x0, y0 - text_size[1], x0 + text_size[0], y0],
				fill=tuple(color))"""
			draw.text((x0, y0 - text_size[1]), text, fill='red',
				font=font)
	min_dist = min(dist_from_reference)
	rect_coordinates = ((400, 815), (780, 711), (525, 636), (211,660), (400, 815))
	if min_dist<=4:
		draw.line(rect_coordinates, fill=(0,0,255), width=5)
	elif min_dist<=15:
		draw.line(rect_coordinates, fill=(0,128,255), width=5)
	else:
		draw.line(rect_coordinates, fill=(0,255,0), width=5)				
	return np.array(img)

def scaling(pts, scale):
	pts_list = []
	for pt in pts:
		x = pt[0]*scale[0]
		y = pt[1]*scale[1]
		tup = (x,y)
		pts_list.append(tup)
	return pts_list
			

"""	
for boxes_dict in boxes_dicts:
	for cls in range(len(class_names)):
		boxes = boxes_dict[cls]
		if np.size(boxes) != 0:
			for box in boxes:
					xy, confidence = box[:4], box[4]
			
def draw_boxes(img_names, boxes_dicts, class_names, model_size):
	Draws detected boxes.
	Args:
		img_names: A list of input images names.
		boxes_dict: A class-to-boxes dictionary.
		class_names: A class names list.
		model_size: The input size of the model.
	Returns:
		None.
	
	for num, img_name, boxes_dict in zip(range(len(img_names)), img_names,
									boxes_dicts):
		img = Image.open(img_name)
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype(font='data/futur.ttf',
						size=(img.size[0] + img.size[1]) // 100)
		resize_factor = \
				(img.size[0] / model_size[0], img.size[1] / model_size[1])
		for cls in range(len(class_names)):
			boxes = boxes_dict[cls]
			if np.size(boxes) != 0:
				color = np.random.permutation([np.random.randint(256), 255, 0])
				for box in boxes:
					xy, confidence = box[:4], box[4]
					xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
					x0, y0 = xy[0], xy[1]
					thickness = (img.size[0] + img.size[1]) // 200
					for t in np.linspace(0, 1, thickness):
						xy[0], xy[1] = xy[0] + t, xy[1] + t
						xy[2], xy[3] = xy[2] - t, xy[3] - t
						draw.rectangle(xy, outline=tuple(color))
					text = '{} {:.1f}%'.format(class_names[cls],
										confidence * 100)
					text_size = draw.textsize(text, font=font)
					draw.rectangle(
						[x0, y0 - text_size[1], x0 + text_size[0], y0],
						fill=tuple(color))
					draw.text((x0, y0 - text_size[1]), text, fill='black',
						font=font)
	img.save("image1.jpeg")
"""


class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    def __init__(self, pixel_array=quad_coords['pixel'], lonlat_array=quad_coords['lonlat']):
        assert pixel_array.shape==(4,2), "Need (4,2) input array"
        assert lonlat_array.shape==(4,2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array),np.float32(pixel_array))
        
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        lonlat = np.dot(self.M,pixel.T)
        
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1,2)
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,lonlat.T)
        
        return (pixel[:2,:]/pixel[2,:]).T
