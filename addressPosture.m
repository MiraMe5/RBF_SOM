function dataimg = addressPosture(data,varargin)%one frame
path1=data{1};

filelist1=dir(path1);%Get the data from current dir
filelist=filelist1(3:end);
cou=1; 
%--------------------------------- get image of train&labels
dataimg=[];labelss=[];
labelss=[];labels1=[];
for hh=1:length(filelist)
    
    file=filelist(hh);
    
    
    path3=fullfile(path1,file.name);
    
    filelist1=dir(path3);%Get the data from current dir
    filelist11=filelist1(3:end);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
        for xx=1:size(filelist11)
            
                %dataimg{cou}=fullfile(file.name,filelist11(xx).name);
                dataimg{cou}=file.name;
                cou=cou+1;
        end
    
    
end
