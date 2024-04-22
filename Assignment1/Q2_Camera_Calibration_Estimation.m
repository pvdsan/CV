% Auto-generated by cameraCalibrator app on 19-Apr-2024
%-------------------------------------------------------


% Define images to process
imageFileNames = {'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image1.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image2.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image3.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image4.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image5.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image6.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image7.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image8.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image9.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image10.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image11.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image12.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image13.png',...
    'C:\Users\Sanket\Desktop\Semester2\CV\Assignment1\Calibration_Images\Image14.png',...
    };
% Detect calibration pattern in images
detector = vision.calibration.monocular.CheckerboardDetector();
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates for the planar pattern keypoints
squareSize = 25;  % in units of 'millimeters'
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);
% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')
