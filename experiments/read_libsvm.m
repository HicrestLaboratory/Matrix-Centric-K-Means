function features = read_libsvm(file_path, n, d)
    % Open the file
    fid = fopen(file_path, 'r');
    
    if fid == -1
        error('Cannot open file: %s', file_path);
    end
    
    % Initialize arrays to store labels and features
    labels = [];
    features = zeros(n, d);
    
    line_num = 1;

    % Read the file line by line
    line = fgetl(fid);
    while ischar(line)

        fprintf("Reading line %d\n", line_num)


        % Split the line into tokens
        tokens = strsplit(line);
        
        % First token is the label
        label = str2double(tokens{1});
        
        % Remaining tokens are the features
        for i = 2:length(tokens)
            feature = strsplit(tokens{i}, ':');
            if size(feature)<2
                continue;
            end
            index = str2double(feature{1});
            value = str2double(feature{2});
            features(line_num, index) = value;
        end

        line_num = line_num + 1;
        
        % Append to the labels and features arrays
        % labels = [labels; label];

        % features = [features; feature_vector];
        
        % Read the next line
        line = fgetl(fid);
    end
    
    % Close the file
    fclose(fid);
end
