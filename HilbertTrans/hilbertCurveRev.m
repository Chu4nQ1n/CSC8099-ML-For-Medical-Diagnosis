function outputData = hilbertCurveRev(data)
%   Summary of this function goes here
%   Detailed explanation goes here

rowNumel = sqrt(numel(data));
order = log(numel(data))/log(4);

% convert original index to x,y format
a = 1 + 1i;
b = 1 - 1i;

% Generate point sequence
z = 0;
for k = 1:order
    w = 1i*conj(z);
    z = [w-a; z-b; z+a; b-w]/2;
end

newCol = real(z);
newRow = imag(z);
newCol = rowNumel*newCol/2 + rowNumel/2 + 0.5;
newRow = rowNumel*newRow/2 + rowNumel/2 + 0.5;
hilbertInd = (newCol-1)*rowNumel+newRow;
outputData(hilbertInd) = data;
outputData = reshape(outputData,rowNumel,rowNumel);

end