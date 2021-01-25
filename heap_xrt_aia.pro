; stack XRT and 6-channels AIA data as the input of aia2xrt_train/test.py or xrt1024_generator.py

f=file_search('com*.fits',count=n)
ft=file2time(f)
f171=findfile('*_0171.*') & ft171=file2time(f171)
f193=findfile('*_0193.*') & ft193=file2time(f193)
f211=findfile('*_0211.*') & ft211=file2time(f211)
f335=findfile('*_0335.*') & ft335=file2time(f335)
f131=findfile('*_0131.*') & ft131=file2time(f131)
f94=findfile('*_0094.*') & ft94=file2time(f94)

for i=0,n-1 do begin 
   fits2map,f[i],xmap,header=hdr
   xmap.dx=xmap.dx*0.9972 & xmap.dy=xmap.dy*0.9972
   xr=get_map_xrange(xmap) & yr=get_map_yrange(xmap) & dim=get_map_dim(xmap)
   j1=(near_time(ft171,ft[i]))[0]
   j2=(near_time(ft193,ft[i]))[0]
   j3=(near_time(ft211,ft[i]))[0]
   j4=(near_time(ft335,ft[i]))[0]
   j5=(near_time(ft131,ft[i]))[0]
   j6=(near_time(ft94,ft[i]))[0]
   faia=[f171[j1],f193[j2],f211[j3],f335[j4],f131[j5],f94[j6]]
   faiat=[ft171[j1],ft193[j2],ft211[j3],ft335[j4],ft131[j5],ft94[j6]]
   dt=abs(anytim2tai(faiat)-anytim2tai(ft[i]))
   print,'===================='
   print,max(dt)
   if max(dt) gt 12 then continue
   fits2map,faia,amap
   amap=get_sub_map(amap,xra=xr,yra=yr)
   amap=rebin_map(amap,dim[0],dim[1])
   for j=0,5 do amap[j]=inter_map(amap[j],xmap)
   ;xmap=get_sub_map(xmap,xr=[-1000,1000],yr=[-1000,1000])
   ;amap=get_sub_map(amap,xr=[-1000,1000],yr=[-1000,1000])
   heapdata=fltarr(dim[0],dim[1],7)
   for j=0,5 do heapdata[*,*,j]=amap[j].data
   heapdata[*,*,6]=xmap.data
   writefits,'heap_'+f[i],heapdata,hdr
   print,'heap_'+f[i]
   file_delete,f[i]
   print,'delete: ',f[i]
endfor
end
