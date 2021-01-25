; stack 6-channels AIA images as the input of xrt4096_generator.py

f335=file_search('*_0335.*',count=n) & ft=file2time(f335)
f171=findfile('*_0171.*') & ft171=file2time(f171)
f193=findfile('*_0193.*') & ft193=file2time(f193)
f211=findfile('*_0211.*') & ft211=file2time(f211)
f131=findfile('*_0131.*') & ft131=file2time(f131)
f94=findfile('*_0094.*') & ft94=file2time(f94)
heapdata=fltarr(4096,4096,6)
for i=0,n-1 do begin 
 j1=(near_time(ft171,ft[i]))[0]
 j2=(near_time(ft193,ft[i]))[0]
 j3=(near_time(ft211,ft[i]))[0]
 j4=(near_time(ft131,ft[i]))[0]
 j5=(near_time(ft94,ft[i]))[0]
 faiat=[ft171[j1],ft193[j2],ft211[j3],ft131[j4],ft94[j5]]
 dt=abs(anytim2tai(faiat)-anytim2tai(ft[i]))
 if max(dt) gt 12 then continue
 faia=[f171[j1],f193[j2],f211[j3],f335[i],f131[j4],f94[j5]]
 fits2map,faia,maps,header=hdr
 heapname='heap6_aia_'+time2file(ft[i],/sec)+'.fits'
 ;map2fits,maps,heapname
 for j=0,5 do heapdata[*,*,j]=maps[j].data
 writefits,heapname,heapdata,hdr
 print,i,' /',n-1
 print,'write fits: ',heapname
endfor
end
