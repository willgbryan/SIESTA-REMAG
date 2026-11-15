// Execute the shipped client against a minimal DOM. This is not visual QA.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
let state=JSON.parse(fs.readFileSync(process.argv[3],'utf8')),posted=[],refuse=false;
class Element {
  constructor(){this.children=[];this.value='';this.checked=false;this.style={};this.hidden=false;this.textContent='';this.dataset={};}
  append(...children){this.children.push(...children);if(!this.value&&children[0]?.value)this.value=children[0].value;}
  replaceChildren(...children){this.children=[];this.value='';this.append(...children);}
  setAttribute(name,value){this[name]=value;}
  toggleAttribute(name,value){this[name]=value;}
  getBoundingClientRect(){return {width:800,height:600,left:0,top:0};}
  getContext(){return new Proxy({},{get:(_,key)=>key==='createRadialGradient'?()=>({addColorStop(){}}):()=>{}});}
}
const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>['#'+m[1],new Element()]));
for(const name of ['h1','.stage-label','.stage-bottom','.axis','.legend'])elements.set(name,new Element());
elements.get('#author-operation').value='apply';elements.get('#query-mode').value='query';elements.get('#query-kind').value='scalar';elements.get('#query-label').value='New experiment';elements.get('#series-control').value='strain-2';
const context={document:{querySelector:s=>elements.get(s)||null,querySelectorAll:()=>[],createElement:()=>new Element(),createElementNS:()=>new Element(),createTextNode:text=>({textContent:text})},ResizeObserver:class{observe(){}},devicePixelRatio:1,window:{devicePixelRatio:1},setTimeout,clearTimeout,console,Blob,URL,
 fetch:async(path,options)=>{
  if(path==='/api/state')return {ok:true,json:async()=>structuredClone(state)};
  assert.equal(path,'/api/prepare');assert.equal(options.method,'POST');const payload=JSON.parse(options.body);posted.push(payload);
  if(refuse)return {ok:false,json:async()=>({ok:false,error:{message:'Source or calculation settings changed; refresh the workspace'}})};
  const entry={id:'prepared-'+posted.length,label:payload.label,kind:payload.kind,source_sha256:payload.expected_source_sha256};
  state.research.push(entry);return {ok:true,json:async()=>({ok:true,result:{entry,calculation_performed:false}})};
 }};
vm.createContext(context);vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const run=code=>vm.runInContext(code,context);
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(elements.get('#error').textContent,'');assert.equal(elements.get('#query-controls').hidden,false);
 elements.get('#selection-orbit').value='@coordination';elements.get('#selection-orbit').onchange({target:elements.get('#selection-orbit')});
 assert.equal(elements.get('#motif-scope').hidden,false);assert.deepEqual(Array.from(run('selectedAtoms()')).sort(),[0,1]);
 assert.equal(run('selectedAtoms().length'),2); // Multiple periodic neighbor images must not duplicate orbital weights.
 assert.equal(elements.get('#saved-responses').hidden,false);
 assert.equal(elements.get('#response-points').children.length,3);
 assert.equal(elements.get('#response-points').children[2].children[3].textContent,'Branch held');
 assert.match(elements.get('#response-scope').textContent,/unknown/);
 assert.match(elements.get('#response-derivatives').children[0].textContent,/Unavailable/);
 elements.get('#response-observable').value='K';elements.get('#response-observable').onchange();
 assert.equal(elements.get('#response-observable').value,'K');assert.match(elements.get('#response-definition').textContent,/anisotropy/);
 await elements.get('#prepare-query').onclick();
 assert.equal(posted.length,1);assert.equal(posted[0].source_id,'base');assert.deepEqual(posted[0].query,{kind:'scalar'});
 assert.equal(posted[0].expected_profile_sha256,state.research[0].sha256);
 assert.equal(elements.get('#prepare-query').disabled,false);assert.match(elements.get('#query-status').textContent,/No calculation started/);
 elements.get('#query-mode').value='series';elements.get('#query-mode').onchange();assert.equal(elements.get('#series-controls').hidden,false);
 elements.get('#series-values').value='-.01,0,.01';await elements.get('#prepare-query').onclick();assert.equal(posted.length,1);
 elements.get('#series-values').value='-1/200, 0, 1/100';await elements.get('#prepare-query').onclick();
 assert.deepEqual(posted[1].design,{format:'REMAG-RESPONSE-DESIGN/1',parameter:'q',values:['-1/200','0','1/100'],control:{kind:'axial_strain',axis:2}});
 elements.get('#series-control').value='metric_path';elements.get('#series-control').onchange();assert.equal(elements.get('#series-metric').hidden,false);
 elements.get('#series-parameter').value='metric.g33';elements.get('#series-rate').value='2/3';await elements.get('#prepare-query').onclick();
 assert.deepEqual(posted[2].design.control,{kind:'metric_path',rates:{'metric.g33':'2/3'}});
 refuse=true;await elements.get('#prepare-query').onclick();assert.match(elements.get('#query-status').textContent,/changed/);assert.equal(elements.get('#prepare-query').disabled,false);
 const count=posted.length;elements.get('#observed').checked=true;run('updateResearch()');assert.equal(elements.get('#query-controls').hidden,true);await elements.get('#prepare-query').onclick();assert.equal(posted.length,count);
 elements.get('#observed').checked=false;run('data.authoring_enabled=false;updateResearch()');assert.equal(elements.get('#query-controls').hidden,true);
 run("data.authoring_enabled=true;data.research=data.research.filter(e=>e.kind!=='profile');updateResearch()");assert.equal(elements.get('#prepare-query').disabled,true);
 elements.set('#saved-state',new Element());run('updateResearch()');assert.equal(elements.get('#query-controls').hidden,true);await elements.get('#prepare-query').onclick();assert.equal(posted.length,count);
 run("data.families[0].rows[0].structure.source.sha256='e'.repeat(64);updateResearch()");assert.equal(elements.get('#saved-responses').hidden,true);assert.equal(elements.get('#response-points').children.length,0);
 console.log('Workspace preparation and response transitions passed; visual rendering not assessed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
