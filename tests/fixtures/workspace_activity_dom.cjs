const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
class Element {
 constructor(){this.children=[];this.hidden=false;this.value='';this.textContent='';}
 append(...children){this.children.push(...children)}
 replaceChildren(){this.children=[];this.value=''}
}
const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>['#'+m[1],new Element()]));
let response={available:true,title:'Astra',status:'Stopped',summary:'Original failure retained',next:'Diagnose',updated_at:'2026-11-16',stages:[],policies:[{label:'Physics',status:'Sealed',actions:[{sequence:'000',tool:'record_hypothesis',status:'passed',rationale:'<script>must remain text</script>',record:{statement:'Public note'}}]}]},fail=false,timer,fetches=0;
const context={document:{createElement:()=>new Element()},$:(s)=>elements.get(s)||null,data:{activity_enabled:true},window:{location:{hash:''}},clearTimeout:()=>{},setTimeout:cb=>{timer=cb},fetch:async path=>{assert.equal(path,'/api/activity');fetches++;return {ok:!fail,json:async()=>response}}};
vm.createContext(context);
const code=html.slice(html.indexOf('let activity=null'),html.indexOf('canvas.onpointerdown='));
vm.runInContext(code,context);
const run=code=>vm.runInContext(code,context);
(async()=>{
 await run('loadActivity()');assert.equal(elements.get('#activity-status').textContent,'Stopped');assert.equal(fetches,1);assert.equal(typeof timer,'function');
 const item=elements.get('#activity-actions').children[0];assert.equal(item.children[2].textContent,'<script>must remain text</script>');assert.equal(item.children[2].innerHTML,undefined);
 // Unchanged polling must preserve expanded user reading state.
 item.open=true;await timer();assert.equal(elements.get('#activity-actions').children[0],item);assert.equal(item.open,true);
 response={...response,status:'Diagnostic build running'};await timer();assert.equal(elements.get('#activity-status').textContent,'Diagnostic build running');
 // Receipt directory identifiers are opaque (live phases include a name prefix).
 response.policies[0].actions.push({...response.policies[0].actions[0],sequence:'iteration-1-000'});
 await timer();assert.equal(elements.get('#activity-actions').children[0].children[0].textContent,'1. record hypothesis · passed');
 assert.equal(elements.get('#activity-actions').children[1].children[0].textContent,'2. record hypothesis · passed');
 fail=true;await timer();assert.match(elements.get('#activity-connection').textContent,/Disconnected.*stale/);assert.equal(elements.get('#activity-status').textContent,'Diagnostic build running');
 // Snapshots are explicitly static and never fetch or schedule updates.
 elements.set('#saved-state',new Element());run('data.activity=activity');const calls=fetches;timer=null;await run('loadActivity()');assert.equal(fetches,calls);assert.equal(timer,null);assert.match(elements.get('#activity-connection').textContent,/Saved snapshot/);
 console.log('Activity refresh, stale/snapshot states, literal text and reading-state preservation passed.');
})().catch(e=>{console.error(e);process.exitCode=1});
