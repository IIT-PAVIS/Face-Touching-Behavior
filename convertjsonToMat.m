function convertjsonToMat(path1, allJsonFile_names,destination)


    currAvi=allJsonFile_names;
    currAvi_woutExt=currAvi(1:end-5);
    curr_file=[path1 currAvi];
    fid = fopen(curr_file);
    tline = fgetl(fid);
    i=1;
    [data(i) json] = parse_json(tline); 
    %Download the code in: https://it.mathworks.com/matlabcentral/fileexchange/20565-json-parser
    %to use the parse_json function
    i=i+1;
    while ischar(tline) || tline~=-1
        disp(tline)
        tline = fgetl(fid);
        if tline~=-1
            [data(i) json] = parse_json(tline);
            i=i+1;
        end
    end
    fclose(fid);
    save(destination,'data');

end
