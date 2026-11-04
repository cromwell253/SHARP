const fs=require('fs'),path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE_PATH||'playwright');
(async()=>{
 const out=path.resolve(process.argv[2]),ready=JSON.parse(fs.readFileSync(path.join(out,'backend_ready.json'),'utf8'));
 const browser=await chromium.launch({headless:true,channel:process.platform==='win32'?'msedge':undefined,args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});
 const page=await browser.newPage({viewport:{width:1320,height:620},deviceScaleFactor:1});await page.goto(`http://127.0.0.1:${ready.port}`);await page.evaluate(()=>window.init());
 const summary={protocol:ready,browser:browser.version(),paint_completion:'Two requestAnimationFrame callbacks after canvas draw; headless browser compositor.',models:{}};
 const quantile=(a,q)=>{const p=(a.length-1)*q,i=Math.floor(p);return a[i]+(a[Math.ceil(p)]-a[i])*(p-i)};
 for(const model of ['SHARP','PatchSTG']){
  const samples=await page.evaluate(m=>window.runTest(m),model);fs.writeFileSync(path.join(out,`${model}_ui_latencies.csv`),'index,end_to_end_ms,model_ms\n'+samples.map(x=>`${x.index},${x.end_to_end_ms},${x.model_ms}`).join('\n'));
  const a=samples.map(x=>x.end_to_end_ms).sort((x,y)=>x-y);summary.models[model]={updates:a.length,p50_ms:quantile(a,.5),p95_ms:quantile(a,.95),p99_ms:quantile(a,.99),max_ms:a.at(-1),deadline_misses:a.filter(x=>x>60000).length};
  await page.screenshot({path:path.join(out,`${model}_prototype.png`)});fs.writeFileSync(path.join(out,'visual_pipeline_summary.json'),JSON.stringify(summary,null,2));console.log(model,JSON.stringify(summary.models[model]));
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1});
