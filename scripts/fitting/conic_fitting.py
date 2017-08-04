"""
author: fangren
"""

from scipy.optimize import least_squares

from scripts.clustering.clustering import clustering
from scripts.fitting.xy_to_Qchi import *
from scripts.importData.create_fake_image import create_fake_image

if __name__ == '__main__':
    # clustering
    img_path = 'LaB6//3.tif'
    n_clusters = 15
    labels, X_coordinates, Y_coordinates= clustering(img_path, n_clusters)
    img, im_simple, im_adaptive, im_combine = import_image(img_path)
    # plot one of the arcs picked up by clustering, defined by the label number
    label_num = 14
    X = X_coordinates[labels == label_num]
    Y = Y_coordinates[labels == label_num]
    plt.scatter(X, Y, alpha = 0.5)
    # import parameters true parameters  for validation
    params = parse_calib('LaB6//3.calib')
    print 'The true geometric parameters are', params

    # guess values and bounds for fitting
    guess = [280, 4.72, 0.539, 103, 231]
    #guess = [200, 4.62, 0.439, 3, 131]
    low = [200, 4.62, 0.439, 3, 131]
    high = [380, 4.82, 0.639, 203, 331]

    def remeshing(params):
        detector_dist, detect_tilt_alpha, detect_tilt_delta, bcenter_x, bcenter_y = params
        wavelength = 0.9762
        Q = np.zeros_like(X)
        chi = np.zeros_like(X)
        Q[:], chi[:] = Qchi(X, Y, np.concatenate((params, [wavelength])))
        return Q, chi

    def object_func(params):
        Q, chi = remeshing(params)
        J = sum((Q - 2.12) ** 2)
        return J

    fitting_result = least_squares(object_func, guess)
    params_fitted =  fitting_result.x
    # create the fake image according to the parameters
    shape = (205, 205)
    pixelSize = 79
    # create fake image
    img_fitted = create_fake_image(np.concatenate((params_fitted, [0.9762])), pixelSize, shape)
    print 'The fitted geometric parameters are', np.concatenate((params_fitted, [0.9762]))
    # visualize the fake image and the original data points for fitting

    X = X_coordinates
    Y = Y_coordinates
    Q, chi = remeshing(params_fitted)
    plt.figure(1)
    plt.subplot(121)
    plt.scatter(Q, chi)
    plt.subplot(122)
    plt.imshow(img_fitted, alpha = 0.7)
    plt.imshow(img, alpha = 0.7)
    plt.show()
