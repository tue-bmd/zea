function new_filename = unique_filename(directory, base_filename, extension)
    % UNIQUE_FILENAME Generate a unique filename in the specified directory
    %
    %   new_filename = UNIQUE_FILENAME(directory, base_filename, extension) generates
    %   a unique filename in the specified directory with the specified base filename
    %   and file extension. The function appends a 4-digit number to the base filename,
    %   incrementing the number until a non-existing filename is found.
    %
    %   Inputs:
    %       directory     - The directory in which to check for existing filenames.
    %                       The directory should be specified as a string.
    %       base_filename - The base filename (without extension) to use for generating
    %                       the new filename. The base filename should be specified as a string.
    %       extension     - The file extension to use for the new filename. The extension
    %                       should be specified as a string, with or without a leading dot.
    %
    %   Output:
    %       new_filename  - The generated unique filename, including the full path,
    %                       constructed as:
    %                       [directory, base_filename, '_', 4-digit number, extension].

    % Ensure the directory ends with a file separator
    if directory(end) ~= filesep
        directory = [directory, filesep];
    end

    % Ensure the extension starts with a dot
    if ~startsWith(extension, '.')
        extension = ['.', extension];
    end

    % Initialize a counter for the file number
    file_number = 0;

    % Generate the initial filename with a 4-digit number
    new_filename = sprintf('%s%s_%04d%s', directory, base_filename, file_number, extension);

    % Increment the file number until a non-existing filename is found
    while exist(new_filename, 'file') == 2
        file_number = file_number + 1;
        new_filename = sprintf('%s%s_%04d%s', directory, base_filename, file_number, extension);
    end
end
