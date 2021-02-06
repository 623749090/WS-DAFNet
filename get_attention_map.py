import numpy as np
import cv2


def get_au_tg_dlib(array_68, h, w):
	
	str_dt = list(array_68[:, 0])+list(array_68[:, 1])
	region_array = np.zeros((11, 4))
	try:
		W = w
		H = h
		arr2d = np.array(str_dt).reshape((2, 68))
		# print arr2d
		arr2d[0, :] = arr2d[0, :]/W*100
		arr2d[1, :] = arr2d[1, :]/H*100
		region_bbox = []
		ruler = abs(arr2d[0, 39]-arr2d[0, 42])
		# print ruler
		region_bbox += [[arr2d[0, 21], arr2d[1, 21]-ruler/2, arr2d[0, 22], arr2d[1, 22]-ruler/2]] #0
		region_bbox += [[arr2d[0, 18], arr2d[1, 18]-ruler/3, arr2d[0, 25], arr2d[1, 25]-ruler/3]] #2
		
		region_bbox += [[arr2d[0, 19], arr2d[1, 19]+ruler/3, arr2d[0, 24], arr2d[1, 24]+ruler/3]] #3: au4
		region_bbox += [[arr2d[0, 41], arr2d[1, 41]+ruler, arr2d[0, 46], arr2d[1, 46]+ruler]] #4: au6
		region_bbox += [[arr2d[0, 38], arr2d[1, 38], arr2d[0, 43], arr2d[1, 43]]] #5: au7
		region_bbox += [[arr2d[0, 49], arr2d[1, 49], arr2d[0, 53], arr2d[1, 53]]] #6: au10
		region_bbox += [[arr2d[0, 48], arr2d[1, 48], arr2d[0, 54], arr2d[1, 54]]] #7: au12 au14 lip corner
		region_bbox += [[arr2d[0, 51], arr2d[1, 51], arr2d[0, 57], arr2d[1, 57]]] #8: au17
		region_bbox += [[arr2d[0, 61], arr2d[1, 61], arr2d[0, 63], arr2d[1, 63]]] #9: au 23 24
		region_bbox += [[arr2d[0, 56], arr2d[1, 56]+ruler/2, arr2d[0, 58], arr2d[1, 58]+ruler/2]] #10: #au23

		region_array = np.array(region_bbox)
	except Exception as e:
		i = 0
	return region_array


def get_map_single_au(array_68, h, w):
	feat_map = np.zeros((100, 100))
	tg_array = get_au_tg_dlib(array_68, h, w)
	for i in range(tg_array.shape[0]):	 # in range(10)
		for j in range(2):
			pt = tg_array[i, j*2:(j+1)*2]
			pt = pt.astype('uint8')
			for px in range(pt[0]-5, pt[0]+5):
				if px < 0 or px > 99:
					break
				for py in range(pt[1]-5, pt[1]+5):
					if py < 0 or py > 99:
						break
					d1 = abs(px-pt[0])
					d2 = abs(py-pt[1])
					value = 1-(d1+d2)*0.095
					feat_map[py][px] = max(feat_map[py][px], value)
	return feat_map
