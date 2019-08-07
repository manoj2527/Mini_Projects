matlabroot = '/usr/local/MATLAB/R2019a';

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

load net.mat;
rng(0);
imds = shuffle(imds);
ni = 12;
imds.ReadSize = ni;
pure = transform(imds,@commonPreprocessing);
T = read(pure);
dsTest = transform(imds,@addNoise);
dsTest = transform(dsTest,@commonPreprocessing);
T1 = read(dsTest);
ypred = predict(net,dsTest);
T2 = ypred(:,:,:,1:ni);
n = 32;
full = zeros(n*ni,n*3);
error = zeros(1,ni);
ssims = zeros(1,ni);
rmse = zeros(1,ni);
for i=0:ni-1
    orig = T{i+1};
    full(n*i+1:n*i+n,1:n) = orig;
    noise = T1{i+1};
    full(n*i+1:n*i+n,n+1:2*n) = noise;
    img = T2(:,:,:,i+1);
    full(n*i+1:n*i+n,2*n+1:3*n) = img;
    error(1,i+1) = sum((img-orig).*(img-orig))/sum(orig.*orig);
    [ssimval, ssimmap] = ssim(double(orig),double(img));
    ssims(1,i+1) = ssimval;
    rmse(1,i+1) = sqrt(sum(sum((img-orig).*(img-orig)))/sum(sum(orig.*orig)));
end
net.Layers
ssims
rmse
imshow(full);
title('original-noise-denoised');
pause(2);
fprintf('denoising completed\n');
%% d(ii)

%layers = [2,6,10,14,16,18,20];
%bounds = [32,16,12,12,16,32,1];

layers = [2,6,10,14,16,18,20];
bounds = [8,16,32,32,16,8,1];

% layers = [2,5,8,11,13,15,17];
% bounds = [32,16,8,8,16,32,1];

for i=1:7
    channels = 1:bounds(i);
    I = deepDreamImage(net,layers(i),channels,'PyramidLevels',1,'Verbose',false);
    name = net.Layers(layers(i)).Name;
    figure
    I = imtile(I,'ThumbnailSize',[128 128]);
    imshow(I)
    title(['Layer ',name,' Features']);
    pause(1);
end
pause(2);
fprintf("weights displayed\n");
%% d(iii)

for i=1:7
    name = net.Layers(layers(i)).Name;
    act1 =  activations(net,T{1},name);
    sz = size(act1);
    figure
    I = imtile(mat2gray(act1),'Gridsize',[ceil(bounds(i)/3),3]);
    imshow(I);
    title(['Layer ',name,' Activations'])
    pause(1);
end
pause(2);
fprintf('activations displayed\n');
%% e(i)

prenet = denoisingNetwork('DnCNN');
full = zeros(n*ni,n*3);
for i=0:ni-1
    noise = T1{i+1};
    img = zeros(64,64);
    img(1:32,1:32) = noise;
    img(33:64,33:64) = noise;
    img(1:32,33:64) = noise;
    img(33:64,1:32) = noise;
    I = denoiseImage(img,prenet);
    img = I(1:32,1:32);
    orig = T{i+1};
    full(n*i+1:n*i+n,1:n) = orig;
    noise = T1{i+1};
    full(n*i+1:n*i+n,n+1:2*n) = noise;
    full(n*i+1:n*i+n,2*n+1:3*n) = img;
    error(1,i+1) = sum((img-orig).*(img-orig))/sum(orig.*orig);
    [ssimval, ssimmap] = ssim(double(orig),double(img));
    ssims(1,i+1) = ssimval;
    rmse(1,i+1) = sqrt(sum(sum((img-orig).*(img-orig)))/sum(sum(orig.*orig)));
    pause(1);
end
ssims
rmse
imshow(full);
title('original-noise-denoised(pre-trained)');
pause(2);
fprintf('denoised with existing network');
%% e(ii)
I = im2double(imread('cameraman.tif'));
img = poissrnd(I*16);
img = img/max(img(:));

for i=1:8
    for j=1:8
        part = zeros(32,32,1);
        part(1:32,1:32,:) = img(32*(i-1)+1:32*i,32*(j-1)+1:32*j);
        ourrec =  predict(net,part);
        rec(32*(i-1)+1:32*i,32*(j-1)+1:32*j) = ourrec;
    end
end

figure;
subplot(1,3,1);
imshow('cameraman.tif');
title('original');
subplot(1,3,2);
imshow(img);
title('noise');
subplot(1,3,3);
imshow(rec);
title('denoised');
fprintf('tested with cameraman.tif\n');