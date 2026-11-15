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
elements.get('#author-operation').value='apply';elements.get('#author-label').value='Candidate';
const context={document:{querySelector:s=>elements.get(s)||null,querySelectorAll:()=>[],createElement:()=>new Element(),
  createElementNS:()=>new Element(),createTextNode:text=>({textContent:text})},ResizeObserver:class{observe(){}},
  devicePixelRatio:1,window:{devicePixelRatio:1},setTimeout,clearTimeout,console,Blob,URL,
  fetch:async(path,options)=>{
    if(path==='/api/state')return {ok:true,json:async()=>structuredClone(state)};
    assert.equal(path,'/api/author');assert.equal(options.method,'POST');assert.equal(options.headers['Content-Type'],'application/json');
    const payload=JSON.parse(options.body);posted.push(payload);
    if(refuse)return {ok:false,json:async()=>({ok:false,error:{message:'Stale parent source hash'}})};
    const family=structuredClone(state.families[0]);family.id='new-candidate';family.label=payload.label;
    family.rows[0].label=payload.label;family.rows[0].structure.source.sha256='c'.repeat(64);
    family.rows[0].calculations=[];family.rows[0].result=null;state.families.push(family);
    return {ok:true,json:async()=>({ok:true,result:{entry:{id:family.id,label:family.label}}})};
  }};
vm.createContext(context);vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const run=code=>vm.runInContext(code,context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(elements.get('#error').textContent,'');
  assert.equal(elements.get('#author-controls').hidden,false);
  const source=state.families[0].rows[0].structure.source.sha256;
  elements.get('#author-operation').value='physical-tetragonal';elements.get('#author-operation').onchange();
  assert.equal(elements.get('#author-value').value,'101/100');
  await elements.get('#save-candidate').onclick();
  assert.equal(posted.length,1);assert.equal(posted[0].expected_source_sha256,source);
  assert.deepEqual(posted[0].request,{op:'physical-tetragonal',scale:'101/100'});
  assert.equal(run('familyId'),'new-candidate');assert.equal(run('current()'),null);
  assert.equal(elements.get('#save-candidate').disabled,false);
  elements.get('#author-value').value='1.01';await elements.get('#save-candidate').onclick();
  assert.equal(posted.length,1);assert.match(elements.get('#author-status').textContent,/exact integer or fraction/);
  refuse=true;elements.get('#author-value').value='102/100';await elements.get('#save-candidate').onclick();
  assert.equal(posted.length,2);assert.equal(posted[1].expected_source_sha256,'c'.repeat(64));
  assert.equal(elements.get('#author-status').textContent,'Stale parent source hash');
  elements.get('#observed').checked=true;run('updateAuthoring()');assert.equal(elements.get('#author-controls').hidden,true);
  await elements.get('#save-candidate').onclick();assert.equal(posted.length,2);
  elements.get('#observed').checked=false;run('data.authoring_enabled=false;updateAuthoring()');
  assert.equal(elements.get('#author-controls').hidden,true);
  elements.set('#saved-state',new Element());run('data.authoring_enabled=true;updateAuthoring()');
  assert.equal(elements.get('#author-controls').hidden,true);
  console.log('Client authoring transitions passed; visual rendering not assessed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
