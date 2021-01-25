;pre-process the AIA raw 4096x4096 data from JSOC

pro aia_process,file,outdir=outdir,delete=delete,noprep=noprep
if ~keyword_set(outdir) then outdir='./sdox'
if ~is_dir(outdir) then file_mkdir,outdir
f=file_search(file,count=n)
for i=0, n-1 do begin 
print,'prepaing: ',f[i]
read_sdo,f[i],index,data,/uncomp_delete
if ~keyword_set(noprep) then aia_prep,index,data,index1,data1,/norm
if keyword_set(noprep) then begin 
index1=index & data1=data
endif
time=index1.date_obs
ymd_hms=time2file(time,/sec)
wv=trim(index1.wavelnth)
pre='sdo_'
if stregex(index1.instrume,'AIA') ne -1 then begin 
pre=pre+'aia_'
wv=strmid(strtrim(long(wv)/10000.,2),2,4)
;exptime=index1.exptime
;data1=aia_intscale(data1,exp=exptime,wave=wv,/bytescale)
endif
if stregex(index1.instrume,'HMI') ne -1 then pre=pre+'hmi_'
writename=concat_dir(outdir,pre+ymd_hms+'_'+wv+'.fits')
index2map,index1,data1,map
map2fits,map,writename
print,'===>',writename
print,'===>', trim(i)+' /'+trim(n),'   finished'
if keyword_set(delete) then begin 
  file_delete,f[i]
  print,'===>',f[i],'==deleted'
endif
endfor
end
