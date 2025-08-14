%%
function exists = h5exists(filename, path)
    try
        info = h5info(filename, path);
        exists = ~isempty(info);            % Exist => True
    catch
        exists = false;                     % Not exist => False
    end
end

