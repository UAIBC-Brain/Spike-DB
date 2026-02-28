% 生成 1x90 二值向量并输出为可复制的一行文本，同时保存到文件并复制到剪贴板
A = zeros(1,90);

% ---------- 设定要置为1的位置 ----------
idx = [4,10,11,12,16,20,21,29,30,31,41,57,58,67,68,69,73,82,90];   % <- 按需修改为你想要的位置
A(idx) = 1;
% -------------------------------------------------

% 如果你想随机选 n 个位置，取消下面注释并注释上面的 idx 设定：
% n = 5;
% idx = randperm(90, n);
% A = zeros(1,90); A(idx) = 1;

% 把 A 转成 "A = [0,1,0,...];" 的字符串（没有空格）
parts = arrayfun(@num2str, A, 'UniformOutput', false);
body = strjoin(parts, ',');
outstr = ['A = [' body '];'];

% 1) 在命令窗口打印（便于直接复制）
fprintf('%s\n', outstr);

% 2) 尝试复制到系统剪贴板（Windows / macOS）
try
    clipboard('copy', outstr);
    fprintf('已复制到剪贴板。\n');
catch
    fprintf('复制到剪贴板失败（远程会话或部分 Linux 环境可能不支持）。\n');
end

% 3) 保存为文本文件（当前工作目录下 A_output.txt）
save_path = fullfile(pwd, 'SR-SZ脑区索引.txt');
fid = fopen(save_path, 'w');
if fid ~= -1
    fprintf(fid, '%s\n', outstr);
    fclose(fid);
    fprintf('已保存到文件：%s\n', save_path);
else
    error('无法创建文件，请检查写权限或路径是否存在。');
end

% （可选）自动用记事本打开文件（Windows），取消下面一行注释即可：
% system(['notepad "' save_path '"']);

