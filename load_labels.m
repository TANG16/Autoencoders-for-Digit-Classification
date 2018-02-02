function one_hot_labels = load_labels(filename)

file = fopen(filename, 'rb');
magic_number = fread(file, 1, 'int32', 0, 'ieee-be');
numLabels = fread(file, 1, 'int32', 0, 'ieee-be');
labels = fread(file, inf, 'unsigned char');
[~, loc] = ismember(labels, unique(labels));
one_hot_labels = ind2vec(loc');

end