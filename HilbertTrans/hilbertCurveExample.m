% toy data

data = reshape(1:256,16,16);

% transform to hilbert curve
transData = hilbertCurve(data);

% reverse transform to 2D
twoDimData = hilbertCurveRev(transData);

% observation: twoDimData is the same as data, so information is retained
% through the process.

%% Hilbert curve is great for reducing the resolution for faster process!

% toy data
% rowLen = 256;
% data = zeros(rowLen,rowLen);
% for x = 1:rowLen
%     for y = 1:rowLen
%         data(x,y) = exp(-(0.125/rowLen)*((x-(rowLen+1)/2)^2+(y-(rowLen+1)/2)^2));
%     end
% end
data = imread("C:\Users\playc\Desktop\archive\COVID-19_Radiography_Dataset\Viral_Pneumonia\train\Viral Pneumonia-1.png");

data = imresize(data, [256, 256]);



% transform to hilbert curve
transData = hilbertCurve(data);

% reduce dimensionality
reduceRatio = 4; % has to be power of 4
transData = downsample(transData,reduceRatio);

% reverse transform to 2D
twoDimData = hilbertCurveRev(transData);

data2 = imresize(data, [128, 128]);

% plot
figure('Position',[100 100 1000 400]);
subplot(1,3,1);
imshow(data);
subplot(1,3,2);
imshow(twoDimData);
subplot(1,3,3);
imshow(data2);


