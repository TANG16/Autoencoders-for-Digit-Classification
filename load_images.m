function images_all = load_images(filename)

file = fopen(filename, 'rb');

magic_number = fread(file, 1, 'int32', 0, 'ieee-be');

numImages = fread(file, 1, 'int32', 0, 'ieee-be');
numRows = fread(file, 1, 'int32', 0, 'ieee-be');
numCols = fread(file, 1, 'int32', 0, 'ieee-be');

images = fread(file, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(file);

% Reshape to rows* columns*number of images
images = reshape(images, size(images, 1),size(images, 2), size(images, 3));

%% trim images
images = images(5:24,5:24,:);
% Convert to double and rescale to [0,1]
images = double(images) ./ 255;

images_all = mat2cell(images, 20, 20, ones(1,numImages));


end