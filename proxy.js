
/**
 * Simple local proxy for Gemini to avoid browser CORS issues.
 * Usage:
 *   cd server
 *   npm install
 *   node proxy.js
 * This starts http://localhost:8787 and exposes POST /gemini
 */
const http = require('http');
const https = require('https');
const url = require('url');

const PORT = process.env.PORT || 8787;

function proxyGemini(req, res){
  let body = '';
  req.on('data', chunk => body += chunk);
  req.on('end', ()=>{
    try{
      const payload = JSON.parse(body||'{}');
      const model = payload.model || 'gemini-2.5-flash';
      const key = payload.key;
      const gBody = payload.body || {};

      if(!key){
        res.writeHead(400, {'Content-Type':'application/json'});
        res.end(JSON.stringify({error:'Missing key'}));
        return;
      }

      const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(key)}`;
      const u = url.parse(endpoint);
      const options = {
        hostname: u.hostname,
        path: u.path,
        method: 'POST',
        headers: {'Content-Type':'application/json'}
      };

      const r = https.request(options, (rr)=>{
        let resp = '';
        rr.on('data', d=> resp += d);
        rr.on('end', ()=>{
          res.writeHead(200, {'Content-Type':'application/json','Access-Control-Allow-Origin':'*'});
          res.end(resp);
        });
      });
      r.on('error', (e)=>{
        res.writeHead(500, {'Content-Type':'application/json','Access-Control-Allow-Origin':'*'});
        res.end(JSON.stringify({error:'Upstream error', detail: e.message}));
      });
      r.write(JSON.stringify(gBody));
      r.end();
    }catch(e){
      res.writeHead(400, {'Content-Type':'application/json','Access-Control-Allow-Origin':'*'});
      res.end(JSON.stringify({error:'Bad request', detail:e.message}));
    }
  });
}

const server = http.createServer((req, res)=>{
  const parsed = url.parse(req.url, true);
  // CORS preflight
  if(req.method === 'OPTIONS'){
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    });
    res.end();
    return;
  }
  if(req.method==='POST' && parsed.pathname==='/server/gemini'){
    proxyGemini(req, res);
  }else{
    res.writeHead(404, {'Content-Type':'application/json','Access-Control-Allow-Origin':'*'});
    res.end(JSON.stringify({error:'Not found'}));
  }
});

server.listen(PORT, ()=>{
  console.log(`Gemini proxy listening on http://localhost:${PORT}`);
});
