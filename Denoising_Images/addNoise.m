function dataOut = addNoise(data)

dataOut = data;
for idx = 1:size(data,1)
    img = poissrnd(im2double(data{idx})*16);
    img = img/max(img(:));
    dataOut{idx} = img;
end
end