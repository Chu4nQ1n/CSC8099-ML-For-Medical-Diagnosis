% reference https://blog.csdn.net/qq_24193303/article/details/102780421

Input_path = 'C:\Users\playc\Desktop\archive\COVID-19_Radiography_Dataset\Viral_Pneumonia\test\';
Output_path = 'C:\Users\playc\Desktop\HilbertTrans\Viral_Pneumonia_hilbert_test_128\';
namelist = dir(strcat(Input_path,'*.png')); % get all png files under this path

len = length(namelist);

for i = 1:len
    name=namelist(i).name;  %namelist(i).name; % get files name
    I=imread(strcat(Input_path, name)); % Images' full pathname
    % trans 299 to 256
    I = im2gray(I);
    I = imresize(I, [256, 256]);
    transData = hilbertCurve(I);
    % reduce dimensionality, trans to 128
    reduceRatio = 4; % has to be power of 4
    transData = downsample(transData,reduceRatio);
    K2 = hilbertCurveRev(transData);
    
    imwrite(K2,[Output_path,'Viral_Pneumonia-',int2str(i+700),'.png']); % save images 
                       
end

