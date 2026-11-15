// Shipped JavaScript protocol checks with DOM/canvas doubles, not visual QA.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8'),state=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
class Element {
 constructor(){this.children=[];this.value='';this.checked=false;this.style={};this.hidden=false;this.textContent='';this.dataset={};}
 append(...children){this.children.push(...children);if(!this.value&&children[0]?.value)this.value=children[0].value;}
 replaceChildren(...children){this.children=[];this.value='';this.append(...children);}
 setAttribute(k,v){this[k]=v;} toggleAttribute(k,v){this[k]=v;}
 getBoundingClientRect(){return {width:800,height:600,left:0,top:0};}
 getContext(){return new Proxy({},{get:(_,k)=>k==='createRadialGradient'?()=>({addColorStop(){}}):()=>{}});}
}
const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>['#'+m[1],new Element()]));
for(const name of ['h1','.stage-label','.stage-bottom','.axis','.legend'])elements.set(name,new Element());
elements.get('#query-mode').value='query';elements.get('#series-control').value='strain-2';
const context={document:{querySelector:s=>elements.get(s)||null,querySelectorAll:()=>[],createElement:()=>new Element(),createElementNS:()=>new Element(),createTextNode:text=>({textContent:text})},ResizeObserver:class{observe(){}},devicePixelRatio:1,window:{devicePixelRatio:1},setTimeout,clearTimeout,console,Blob,URL,
 fetch:async path=>{assert.equal(path,'/api/state');return {ok:true,json:async()=>structuredClone(state)};}};
vm.createContext(context);vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const run=code=>vm.runInContext(code,context),el=id=>elements.get('#'+id);
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));assert.equal(el('error').textContent,'');
 assert.equal(el('reciprocal-selection').hidden,false);assert.match(el('region-scope').textContent,/2 \/ 2/);
 assert.equal(el('k-select').children.length,2);assert.match(el('band-scope').textContent,/energy-ranked/);
 const all=run('regionalDOS(spectrum()).groups[1].density.slice()');
 el('region-center').value='.5,0,0';el('region-width').value='.01,.1,.1';el('apply-region').onclick();
 assert.equal(el('region-error').textContent,'');assert.match(el('region-scope').textContent,/weight 0.750000/);
 assert.equal(el('k-select').children.length,1);assert.equal(el('k-select').value,'1');
 assert.notDeepEqual(Array.from(run('regionalDOS(spectrum()).groups[1].density')),Array.from(all));
 const selected=run('regionalDOS(spectrum()).groups[1].density.slice()');el('region-center').value='1.5,0,0';el('apply-region').onclick();
 assert.deepEqual(Array.from(run('regionalDOS(spectrum()).groups[1].density')),Array.from(selected));
 el('region-width').value='-.1,.1,.1';el('apply-region').onclick();assert.match(el('region-error').textContent,/half widths/);
 el('region-center').value='.25,0,0';el('region-width').value='.01,.1,.1';el('apply-region').onclick();
 assert.equal(el('k-select').children.length,0);assert.equal(el('state-weight').textContent,'—');
 assert.ok(run('regionalDOS(spectrum()).groups.every(g=>g.density.every(x=>x===0))'));
 el('all-k').onclick();assert.equal(el('k-select').children.length,2);
 el('band-path').value='1,0,1';el('apply-path').onclick();assert.match(el('band-scope').textContent,/3 ordered samples/);
 el('band-path').value='1,999';el('apply-path').onclick();assert.match(el('band-scope').textContent,/selected region/);
 run('drawBands()');const point=el('bands').children.find(n=>n.role==='button'&&n['aria-label'].startsWith('k 0, state 0'));
 assert.ok(point);point.onkeydown({key:'Enter',preventDefault(){}});assert.equal(el('k-select').value,'0');assert.equal(el('band-select').value,'0');
 run('site=1;updateStates()');assert.equal(el('state-weight').textContent,'-0.1000');
 const dot=el('reciprocal').children.find(n=>n.role==='button'&&n['aria-label'].startsWith('k 1 '));dot.onclick();assert.equal(el('k-select').value,'1');
 el('reciprocal-plane').value='1,2';el('reciprocal-plane').onchange();assert.ok(el('reciprocal').children.some(n=>n.textContent.includes('b*')));
 el('region-center').value='.5,0,0';el('region-width').value='.01,.1,.1';el('apply-region').onclick();
 run("spectrum().sampled.identity_sha256='different';updateResponse()");assert.match(el('region-scope').textContent,/2 \/ 2/);
 run('current().spectra={};updateResponse()');assert.equal(el('reciprocal-selection').hidden,true);assert.equal(el('state-selection').hidden,true);
 console.log('Band/region selection, periodicity, partial weights, empty regions, keyboard selection and stale-state reset passed; no visual acceptance.');
})().catch(e=>{console.error(e);process.exitCode=1;});
