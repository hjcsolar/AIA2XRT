;pre-process the composite full-Sun XRT data from SCIA

pro xrt_process,fn,outdir=outdir,firsname=firsname,cor_alig=cor_alig
if n_params() le 0 then pr_syntax,'xrt_name,fn,outdir=outdir,firsname=firsname,cor_alig=cor_alig'
case 1 of 
     data_chk(outdir,/string):  ; user supplied explicitly
     else: outdir='./' ;curdir()
endcase
if ~is_dir(outdir) then mk_dir,outdir,/verb
f=findfile(fn)
n=n_elements(f)
if ~keyword_set(firsname) then firsname='xrt'
print,'==========>total files: ',n
;read_xrt,f,index,data,/force
  
for i=0,n-1 do begin 
 read_xrt,f[i],ind,data
 tt=ind.date_obs
 tt=time2file(tt,/sec)
 wave1=ind.EC_FW1_
 wave2=ind.EC_FW2_
 wave1=strjoin(strsplit(wave1,/extract))
 wave2=strjoin(strsplit(wave2,/extract))
 if tag_exist(ind,'lngfname') and tag_exist(ind,'srtfname') then firsname='com_XRT_LS'
 if tag_exist(ind,'lngfname') and tag_exist(ind,'srtfname') and tag_exist(ind,'medfname') then firsname='com_XRT_LMS'
 ;name=file_break(f[i],/no_ext)
 name=firsname+'_'+tt+'_'+wave1+'_'+wave2+'.fits'
 outname=concat_dir(outdir,name)
 
 if keyword_set(cor_alig) then cor_index=xrt_read_coaldb(ind,/aia_cc,calibration_type=type) else cor_index=ind
  data1=rot(data,-cor_index.crota1,/cub)
  cor_index.crota1=0 & cor_index.crota2=0
  write_xrt,cor_index,data1,/make_dir,outdir=outdir,outfile=name
  if name ne f[i] then file_delete,f[i]
  ;spawn,['mv',f[i],outname],/noshell
print,i,' / ',f[i],' --->'
print,outname
endfor

end

