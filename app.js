
/* mROAS Optimizer — All-in-one JS (no external libs) */

// ---------- Utilities ----------
const fmt = {
  num: (x) => (isFinite(x) ? x.toLocaleString(undefined, {maximumFractionDigits: 0}) : "—"),
  usd0: (x) => (isFinite(x) ? `$${x.toLocaleString(undefined, {maximumFractionDigits: 0})}` : "—"),
  usd2: (x) => (isFinite(x) ? `$${x.toLocaleString(undefined, {maximumFractionDigits: 2})}` : "—"),
  pct1: (x) => (isFinite(x) ? `${(100*x).toLocaleString(undefined, {maximumFractionDigits: 1})}%` : "—"),
  pct0: (x) => (isFinite(x) ? `${(100*x).toLocaleString(undefined, {maximumFractionDigits: 0})}%` : "—"),
  x1: (x) => (isFinite(x) ? x.toLocaleString(undefined, {maximumFractionDigits: 1}) : "—"),
  x2: (x) => (isFinite(x) ? x.toLocaleString(undefined, {maximumFractionDigits: 2}) : "—")
};
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const median = arr => {
  const xs = [...arr].sort((a,b)=>a-b);
  const n = xs.length;
  if(n===0) return NaN;
  return n%2 ? xs[(n-1)/2] : 0.5*(xs[n/2-1]+xs[n/2]);
};
const mean = arr => arr.reduce((a,b)=>a+b,0)/Math.max(1,arr.length);
const sum = arr => arr.reduce((a,b)=>a+b,0);
const variance = arr => {
  const m = mean(arr); return mean(arr.map(v=>(v-m)**2));
};
const softplus = z => Math.log(1 + Math.exp(z)); // for non-negatives
const invSoftplus = y => Math.log(Math.exp(y) - 1);

// ---------- Global State ----------
let rawCSV = "";
let headers = [];
let rows = [];
let data = []; // parsed, cleaned objects: {spend, adSales, totalSales, roas, troas, date}
let mapping = {spend:null, adSales:null, totalSales:null, roas:null, troas:null, date:null};
let fitResult = null;
let chartMain, chartMarginal, chartProfit;
let currentSpendValue = null;
let sampleLoaded = false;

// ---------- DOM ----------
const fileInput = document.getElementById('fileInput');
const marginInput = document.getElementById('marginInput');
const useTotalSelect = document.getElementById('useTotalSelect');
const currentSpendMode = document.getElementById('currentSpendMode');
const colSpend = document.getElementById('colSpend');
const colAdSales = document.getElementById('colAdSales');
const colTotalSales = document.getElementById('colTotalSales');
const colROAS = document.getElementById('colROAS');
const colTROAS = document.getElementById('colTROAS');
const colDate = document.getElementById('colDate');
const btnFit = document.getElementById('btnFit');
const resultsGrid = document.getElementById('resultsGrid');
const btnExportCSV = document.getElementById('btnExportCSV');
const btnExportPNG = document.getElementById('btnExportPNG');
const btnLoadSample = document.getElementById('btnLoadSample');
const btnExportSession = document.getElementById('btnExportSession');
const legendMain = document.getElementById('legendMain');
const chartTooltip = document.getElementById('chartTooltip');
const geminiKey = document.getElementById('geminiKey');
const geminiModel = document.getElementById('geminiModel');
const useProxy = document.getElementById('useProxy');
const btnGemini = document.getElementById('btnGemini');
const summaryText = document.getElementById('summaryText');

// ---------- CSV Parsing ----------
function parseCSV(text){
  const lines = text.replace(/\r/g,'').split('\n').filter(l=>l.trim().length>0);
  const header = lines[0].split(',').map(h=>h.trim());
  const rows = lines.slice(1).map(line => {
    // handle commas within quoted fields (simple parser)
    const parts = [];
    let cur = '', inQ = false;
    for(let i=0;i<line.length;i++){
      const ch = line[i];
      if(ch === '"'){ inQ = !inQ; continue; }
      if(ch === ',' && !inQ){ parts.push(cur); cur=''; }
      else { cur += ch; }
    }
    parts.push(cur);
    return parts.map(v=>v.trim());
  });
  return {header, rows};
}

function autodetect(headers){
  // likely names for columns
  const norm = h => h.toLowerCase().replace(/[^a-z0-9]/g,'');
  const hmap = headers.map(h => ({h, n:norm(h)}));
  const find = (cands) => {
    for(const cand of cands){
      const f = hmap.find(x => x.n.includes(cand));
      if(f) return f.h;
    }
    return '';
  };
  return {
    spend: find(['spend','adspend','cost']),
    adSales: find(['adsales','attributedsales','salesfromads','ppcsales']),
    totalSales: find(['totalsales','allsales','grosssales']),
    roas: find(['roas']),
    troas: find(['troas','totalroas']),
    date: find(['date','day','period'])
  };
}

function populateMappingSelectors(){
  const opts = [''].concat(headers);
  const fill = (sel, selected) => {
    sel.innerHTML = '';
    for(const o of opts){
      const el = document.createElement('option');
      el.value = o; el.textContent = o || '—';
      if(o === selected) el.selected = true;
      sel.appendChild(el);
    }
  };
  const guess = autodetect(headers);
  mapping = {...guess};
  fill(colSpend, guess.spend);
  fill(colAdSales, guess.adSales);
  fill(colTotalSales, guess.totalSales);
  fill(colROAS, guess.roas);
  fill(colTROAS, guess.troas);
  fill(colDate, guess.date);
}

function buildData(){
  const idx = {
    spend: headers.indexOf(mapping.spend),
    adSales: headers.indexOf(mapping.adSales),
    totalSales: headers.indexOf(mapping.totalSales),
    roas: headers.indexOf(mapping.roas),
    troas: headers.indexOf(mapping.troas),
    date: headers.indexOf(mapping.date)
  };
  const out = [];
  for(const row of rows){
    const toNum = (v) => {
      const c = v.replace(/[$,%\s]/g,'');
      const num = parseFloat(c);
      return isFinite(num) ? num : NaN;
    };
    let spend = idx.spend>=0 ? toNum(row[idx.spend]) : NaN;
    let adSales = idx.adSales>=0 ? toNum(row[idx.adSales]) : NaN;
    let totalSales = idx.totalSales>=0 ? toNum(row[idx.totalSales]) : NaN;
    let roas = idx.roas>=0 ? toNum(row[idx.roas]) : NaN;
    let troas = idx.troas>=0 ? toNum(row[idx.troas]) : NaN;
    let date = idx.date>=0 ? row[idx.date] : '';

    if(!isFinite(spend)) continue;
    if(!isFinite(adSales) && isFinite(roas)) adSales = roas * spend;
    if(!isFinite(totalSales) && isFinite(troas)) totalSales = troas * spend;

    out.push({
      spend: +spend,
      adSales: isFinite(adSales) ? +adSales : null,
      totalSales: isFinite(totalSales) ? +totalSales : null,
      roas: isFinite(roas) ? +roas : (isFinite(adSales) ? (+adSales)/(+spend) : null),
      troas: isFinite(troas) ? +troas : (isFinite(totalSales) ? (+totalSales)/(+spend) : null),
      date: date || null
    });
  }
  // filter out zero/negative spend
  const cleaned = out.filter(r => r.spend>0 && (r.adSales!=null || r.totalSales!=null));
  return cleaned;
}

// ---------- Modeling ----------
// Two candidate models: Hill & Exponential saturation
const Model = {
  // y = Smax * x^beta / (s50^beta + x^beta)
  hill: {
    name: 'Hill',
    paramNames: (useTotal)=> useTotal ? ['Smax','s50','beta','Baseline'] : ['Smax','s50','beta'],
    unpack: (theta, useTotal) => {
      const Smax = Math.exp(theta[0]);
      const s50  = Math.exp(theta[1]);
      const beta = Math.exp(theta[2]);
      const B    = useTotal ? softplus(theta[3]) : 0;
      return {Smax, s50, beta, B};
    },
    f: (x, pars) => {
      const {Smax,s50,beta} = pars;
      const xb = Math.pow(x, beta);
      const den = Math.pow(s50, beta) + xb;
      return den>0 ? Smax * xb / den : 0;
    },
    df: (x, pars) => {
      const {Smax,s50,beta} = pars;
      if(x<=0) return Infinity; // derivative near 0 for beta<1
      const s50b = Math.pow(s50, beta);
      const xb = Math.pow(x, beta);
      const den = Math.pow(s50b + xb, 2);
      return Smax * beta * s50b * Math.pow(x, beta-1) / den;
    },
    init: (xs, ys, useTotal, totals) => {
      const Smax0 = 1.2 * Math.max(...ys.filter(v=>isFinite(v)));
      const s500 = median(xs);
      const beta0 = 1.3;
      let B0 = 0;
      if(useTotal && totals && totals.length){
        const diffs = totals.map((t,i)=> isFinite(t) && isFinite(ys[i]) ? t - ys[i] : null).filter(v=>v!=null);
        B0 = Math.max(0, Math.min(median(diffs), Math.max(...diffs)));
      }
      return [Math.log(Math.max(1e-6,Smax0)), Math.log(Math.max(1e-6,s500)), Math.log(Math.max(1e-6,beta0))].concat(useTotal?[invSoftplus(Math.max(0,B0))]:[]);
    }
  },

  // y = Smax * (1 - exp(-k x))^theta
  expSat: {
    name: 'Exponential Saturation',
    paramNames: (useTotal)=> useTotal ? ['Smax','k','theta','Baseline'] : ['Smax','k','theta'],
    unpack: (theta, useTotal) => {
      const Smax = Math.exp(theta[0]);
      const k    = Math.exp(theta[1]);
      const th   = Math.exp(theta[2]);
      const B    = useTotal ? softplus(theta[3]) : 0;
      return {Smax, k, th, B};
    },
    f: (x, pars) => {
      const {Smax,k,th} = pars;
      const z = 1 - Math.exp(-k*x);
      return Smax * Math.pow(z, th);
    },
    df: (x, pars) => {
      const {Smax,k,th} = pars;
      const z = 1 - Math.exp(-k*x);
      const dz = Math.exp(-k*x) * k;
      return Smax * th * Math.pow(z, th-1) * dz;
    },
    init: (xs, ys, useTotal, totals) => {
      const Smax0 = 1.2 * Math.max(...ys.filter(v=>isFinite(v)));
      const s500 = median(xs);
      const k0 = Math.log(2) / Math.max(1e-6, s500);
      const th0 = 1.1;
      let B0 = 0;
      if(useTotal && totals && totals.length){
        const diffs = totals.map((t,i)=> isFinite(t) && isFinite(ys[i]) ? t - ys[i] : null).filter(v=>v!=null);
        B0 = Math.max(0, Math.min(median(diffs), Math.max(...diffs)));
      }
      return [Math.log(Math.max(1e-6,Smax0)), Math.log(Math.max(1e-6,k0)), Math.log(Math.max(1e-6,th0))].concat(useTotal?[invSoftplus(Math.max(0,B0))]:[]);
    }
  }
};

// Objective: weighted SSE on adSales and (optionally) totalSales
function objectiveFactory(model, xs, yAds, yTotals, useTotal, lambdaW=0.3){
  const n = xs.length;
  const ybar = mean(yAds);
  return (theta) => {
    const pars = model.unpack(theta, useTotal);
    let sseAd = 0, sseTot = 0, countAd = 0, countTot = 0;
    for(let i=0;i<n;i++){
      const yhat = model.f(xs[i], pars);
      if(isFinite(yAds[i])){ const e = yhat - yAds[i]; sseAd += e*e; countAd++; }
      if(useTotal && isFinite(yTotals[i])){
        const tHat = (pars.B || 0) + model.f(xs[i], pars);
        const e2 = tHat - yTotals[i]; sseTot += e2*e2; countTot++;
      }
    }
    const sse = sseAd/(Math.max(1,countAd)) + lambdaW * (useTotal ? sseTot/Math.max(1,countTot) : 0);
    return sse;
  };
}

// Nelder–Mead optimizer (unconstrained, we parameterize to enforce constraints)
function nelderMead(f, x0, step=0.3, maxIter=800, tol=1e-9){
  const n = x0.length;
  // initial simplex
  let simplex = [x0];
  for(let i=0;i<n;i++){
    const xi = x0.slice();
    xi[i] = xi[i] + step;
    simplex.push(xi);
  }
  let values = simplex.map(s => f(s));
  const alpha=1, gamma=2, rho=0.5, sigma=0.5;

  function sortSimplex(){
    const idx = values.map((v,i)=>[v,i]).sort((a,b)=>a[0]-b[0]).map(x=>x[1]);
    simplex = idx.map(i=>simplex[i]);
    values = idx.map(i=>values[i]);
  }
  function centroid(excludeLast=true){
    const m = excludeLast ? simplex.slice(0,-1) : simplex;
    const c = new Array(n).fill(0);
    for(const s of m){ for(let j=0;j<n;j++) c[j]+=s[j]; }
    for(let j=0;j<n;j++) c[j]/=m.length;
    return c;
  }
  sortSimplex();

  for(let iter=0; iter<maxIter; iter++){
    const c = centroid();
    const worst = simplex[n];
    const bestVal = values[0];
    const worstVal = values[n];

    // convergence check
    const spread = Math.max(...values) - Math.min(...values);
    if(spread < tol) break;

    // reflection
    const xr = c.map((ci,j)=> ci + alpha*(ci - worst[j]));
    const fr = f(xr);
    if(fr < values[0]){
      // expansion
      const xe = c.map((ci,j)=> ci + gamma*(xr[j]-ci));
      const fe = f(xe);
      if(fe < fr){
        simplex[n] = xe; values[n] = fe;
      }else{
        simplex[n] = xr; values[n] = fr;
      }
    }else if(fr < values[n-1]){
      simplex[n] = xr; values[n] = fr;
    }else{
      // contraction
      let xc, fc;
      if(fr < worstVal){
        // outside contraction
        xc = c.map((ci,j)=> ci + rho*(xr[j]-ci));
      }else{
        // inside contraction
        xc = c.map((ci,j)=> ci + rho*(worst[j]-ci));
      }
      fc = f(xc);
      if(fc < worstVal){
        simplex[n] = xc; values[n] = fc;
      }else{
        // shrink
        const best = simplex[0];
        for(let i=1;i<simplex.length;i++){
          simplex[i] = simplex[i].map((si,j)=> best[j] + sigma*(si - best[j]));
          values[i] = f(simplex[i]);
        }
      }
    }
    sortSimplex();
  }
  return {theta: simplex[0], value: values[0]};
}

// Kneedle knee detection on fitted curve
function kneePoint(xs, ys){
  // xs increasing, ys increasing; return x at max distance from the straight line
  if(xs.length<3) return xs[Math.floor(xs.length/2)] || xs[0];
  const x0 = xs[0], y0=ys[0];
  const xN = xs[xs.length-1], yN=ys[ys.length-1];
  const dx=xN-x0, dy=yN-y0;
  let maxD=-1, bestX=xs[0];
  for(let i=0;i<xs.length;i++){
    // distance from point to line
    const num = Math.abs(dy*xs[i] - dx*ys[i] + xN*y0 - yN*x0);
    const den = Math.sqrt(dx*dx + dy*dy);
    const d = num/den;
    if(d>maxD){ maxD=d; bestX=xs[i]; }
  }
  return bestX;
}

// Root find f'(x) = target via bisection on [lo,hi]
function findSpendAtDerivative(df, target, lo, hi, maxIter=120){
  let fLo = df(lo) - target, fHi = df(hi) - target;
  // expand hi if needed
  let tries = 0;
  while(fLo* fHi > 0 && tries < 20){
    hi *= 2;
    fHi = df(hi) - target;
    tries++;
  }
  if(fLo * fHi > 0){
    // couldn't bracket; return 0 if even at x=0 derivative < target; else hi
    return fLo < 0 ? 0 : hi;
  }
  for(let i=0;i<maxIter;i++){
    const mid = 0.5*(lo+hi);
    const fm = df(mid) - target;
    if(Math.abs(fm) < 1e-8) return mid;
    if(fLo * fm < 0){ hi = mid; fHi = fm; }else{ lo = mid; fLo = fm; }
  }
  return 0.5*(lo+hi);
}

// Compute R²
function r2(yTrue, yPred){
  const yy = yTrue.filter(v=>isFinite(v));
  const pp = yPred.filter((v,i)=>isFinite(yTrue[i]));
  const ybar = mean(yy);
  const sse = sum(pp.map((p,i)=> (p - yy[i])**2));
  const sst = sum(yy.map(v=> (v - ybar)**2));
  return 1 - sse/Math.max(1e-12, sst);
}

// ---------- Fitting Pipeline ----------
function fitModels(dataset){
  const xs = dataset.map(d=>d.spend);
  const yAds = dataset.map(d=>d.adSales!=null?d.adSales:NaN);
  const yTotals = dataset.map(d=>d.totalSales!=null?d.totalSales:NaN);

  const useTotalAuto = (useTotalSelect.value === 'yes') || (useTotalSelect.value==='auto' && yTotals.filter(x=>isFinite(x)).length >= Math.floor(0.5*dataset.length));
  const lambdaW = 0.3;

  const candidates = [Model.hill, Model.expSat];
  const fits = [];

  for(const m of candidates){
    const theta0 = m.init(xs, yAds, useTotalAuto, yTotals);
    const obj = objectiveFactory(m, xs, yAds, yTotals, useTotalAuto, lambdaW);
    const sol = nelderMead(obj, theta0, 0.25, 900, 1e-10);
    const pars = m.unpack(sol.theta, useTotalAuto);

    // predictions
    const yhat = xs.map(x => m.f(x, pars));
    const r2Ad = r2(yAds, yhat);
    let yhatTot=[], r2Tot = null;
    if(useTotalAuto){
      yhatTot = xs.map(x => (pars.B||0) + m.f(x, pars));
      r2Tot = r2(yTotals, yhatTot);
    }

    // info criteria
    const k = sol.theta.length;
    const adErrs = yAds.map((y,i)=> isFinite(y)? (yhat[i]-y):null).filter(v=>v!=null);
    const sseAd = sum(adErrs.map(e=>e*e));
    const n = adErrs.length;
    const AIC = n * Math.log(Math.max(1e-12, sseAd/n)) + 2*k;

    fits.push({
      model: m.name,
      params: pars,
      theta: sol.theta,
      useTotal: useTotalAuto,
      r2Ad, r2Tot,
      AIC,
      yhat, yhatTot
    });
  }

  fits.sort((a,b)=> a.AIC - b.AIC);
  const best = fits[0];

  // compute optimal spend via margin threshold
  const margin = parseFloat(marginInput.value)/100;
  const df = (x) => best.model==='Hill' ? Model.hill.df(x, best.params) : Model.expSat.df(x, best.params);
  const target = 1 / Math.max(1e-9, margin);
  const lo = 1e-9, hi = 100 * Math.max(...xs, 1);
  const xOpt = findSpendAtDerivative(df, target, lo, hi);
  const yOpt = (best.model==='Hill'? Model.hill.f(xOpt, best.params) : Model.expSat.f(xOpt, best.params));
  const roasOpt = yOpt / Math.max(1e-9, xOpt);
  const mroasOpt = df(xOpt);
  const profitOpt = margin * yOpt - xOpt;

  // knee point
  const xGrid = [];
  const yGrid = [];
  const gridMax = Math.max(xOpt*1.6, Math.max(...xs)*1.3);
  const N = 160;
  for(let i=0;i<=N;i++){
    const x = gridMax * i / N;
    const y = (best.model==='Hill'? Model.hill.f(x, best.params) : Model.expSat.f(x, best.params));
    xGrid.push(x); yGrid.push(y);
  }
  const xKnee = kneePoint(xGrid, yGrid);

  // current spend
  let current = null;
  if(currentSpendMode.value==='last'){
    current = dataset[dataset.length-1].spend;
  }else{
    const k = 7;
    const lastK = dataset.slice(-k).map(d=>d.spend);
    current = currentSpendMode.value==='mean_last_7' ? mean(lastK) : median(lastK);
  }
  currentSpendValue = current;

  const yCur = (best.model==='Hill'? Model.hill.f(current, best.params) : Model.expSat.f(current, best.params));
  const roasCur = yCur / Math.max(1e-9, current);
  const mroasCur = df(current);
  const profitCur = margin * yCur - current;

  // elasticity
  const elasticity = (x)=> {
    const y = (best.model==='Hill'? Model.hill.f(x, best.params) : Model.expSat.f(x, best.params));
    const yp = df(x);
    return y>0 && x>0 ? (yp * x / y) : NaN;
  };

  // headroom
  const spendHeadroom = xOpt - current;
  const roasHeadroom = roasOpt - roasCur;
  const adSalesHeadroom = yOpt - yCur;
  const profitHeadroom = profitOpt - profitCur;

  return {
    best,
    xs, yAds, yTotals,
    xOpt, yOpt, roasOpt, mroasOpt, profitOpt,
    xKnee,
    current, yCur, roasCur, mroasCur, profitCur,
    spendHeadroom, roasHeadroom, adSalesHeadroom, profitHeadroom,
    elasticityCur: elasticity(current),
    elasticityOpt: elasticity(xOpt),
    grid: {x: xGrid, y: yGrid}
  };
}

// ---------- Charts (Custom Canvas) ----------
function clearCanvas(ctx){ ctx.clearRect(0,0,ctx.canvas.width, ctx.canvas.height); }

function plotScatterLine(canvas, points, line, markers, options){
  const ctx = canvas.getContext('2d');
  clearCanvas(ctx);
  const w = canvas.width, h = canvas.height;
  const pad = {l:70,r:20,t:20,b:50};
  const X = p => pad.l + (p - xmin)/(xmax-xmin) * (w - pad.l - pad.r);
  const Y = p => h - pad.b - (p - ymin)/(ymax-ymin) * (h - pad.t - pad.b);

  // compute bounds
  const xs = points.map(p=>p.x).concat(line.map(p=>p.x));
  const ys = points.map(p=>p.y).concat(line.map(p=>p.y));
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = 0, ymax = Math.max(...ys);
  if(xmax<=xmin) xmax = xmin+1;
  if(ymax<=ymin) ymax = ymin+1;

  // axes
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, h-pad.b); ctx.lineTo(w-pad.r, h-pad.b); // x axis
  ctx.moveTo(pad.l, h-pad.b); ctx.lineTo(pad.l, pad.t); // y axis
  ctx.stroke();

  // ticks
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = '12px system-ui';
  const ticks = 5;
  for(let i=0;i<=ticks;i++){
    const tx = xmin + i*(xmax-xmin)/ticks;
    const ty = ymin + i*(ymax-ymin)/ticks;
    // x
    const xx = pad.l + i*(w - pad.l - pad.r)/ticks;
    ctx.fillText(`$${tx.toLocaleString()}`, xx-12, h-pad.b+18);
    // y
    const yy = h - pad.b - i*(h - pad.t - pad.b)/ticks;
    ctx.fillText(`$${ty.toLocaleString()}`, 6, yy+4);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    ctx.moveTo(pad.l, yy); ctx.lineTo(w-pad.r, yy); // y grid
    ctx.stroke();
  }

  // line
  ctx.strokeStyle = '#9b8cff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for(let i=0;i<line.length;i++){
    const p = line[i];
    const x = X(p.x), y = Y(p.y);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  // scatter
  ctx.fillStyle = '#6ee7ff';
  for(const p of points){
    const x = X(p.x), y = Y(p.y);
    ctx.beginPath();
    ctx.arc(x,y,3,0,2*Math.PI);
    ctx.fill();
  }

  // markers
  markers.forEach(m => {
    ctx.strokeStyle = m.color || '#39d98a';
    ctx.lineWidth = 1.5;
    // vertical line
    const x = X(m.x);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    // dot on line
    const y = Y(m.y);
    ctx.fillStyle = m.color || '#39d98a';
    ctx.beginPath(); ctx.arc(x,y,4,0,2*Math.PI); ctx.fill();
    // label
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.font = '12px system-ui';
    ctx.fillText(m.label, Math.min(x+6, w-120), Math.max(y-8, pad.t+12));
  });

  return {scaleX: X, scaleY:Y, bounds:{xmin,xmax,ymin,ymax}, pad};
}

function plotLine(canvas, x, y, xLabel="$ Spend", yLabel="$ Value", markers=[], opts={}){
  const ctx = canvas.getContext('2d');
  clearCanvas(ctx);
  const w = canvas.width, h = canvas.height;
  const pad = {l:70,r:20,t:20,b:50};
  const xmin = Math.min(...x), xmax = Math.max(...x);
  const ymin = Math.min(0, ...y), ymax = Math.max(...y);
  const X = p => pad.l + (p - xmin)/(xmax-xmin) * (w - pad.l - pad.r);
  const Y = p => h - pad.b - (p - ymin)/(ymax-ymin) * (h - pad.t - pad.b);

  // axes
  ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, h-pad.b); ctx.lineTo(w-pad.r, h-pad.b);
  ctx.moveTo(pad.l, h-pad.b); ctx.lineTo(pad.l, pad.t);
  ctx.stroke();

  // ticks
  ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.font = '12px system-ui';
  const ticks = 5;
  for(let i=0;i<=ticks;i++){
    // x
    const tx = xmin + i*(xmax-xmin)/ticks;
    const xx = pad.l + i*(w - pad.l - pad.r)/ticks;
    ctx.fillText(`$${tx.toLocaleString()}`, xx-12, h-pad.b+18);
    // y
    const ty = ymin + i*(ymax-ymin)/ticks;
    const yy = h - pad.b - i*(h - pad.t - pad.b)/ticks;
    const label = opts.yfmt ? opts.yfmt(ty) : ty.toLocaleString();
    ctx.fillText(`${label}`, 6, yy+4);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    ctx.moveTo(pad.l, yy); ctx.lineTo(w-pad.r, yy);
    ctx.stroke();
  }

  // line
  ctx.strokeStyle = opts.line || '#9b8cff'; ctx.lineWidth = 2;
  ctx.beginPath();
  for(let i=0;i<x.length;i++){
    const xx = X(x[i]), yy = Y(y[i]);
    if(i===0) ctx.moveTo(xx,yy); else ctx.lineTo(xx,yy);
  }
  ctx.stroke();

  // markers
  markers.forEach(m => {
    ctx.strokeStyle = m.color || '#39d98a'; ctx.lineWidth = 1.5;
    const xx = X(m.x);
    const yy = Y(m.y);
    ctx.beginPath(); ctx.moveTo(xx, pad.t); ctx.lineTo(xx, h - pad.b); ctx.stroke();
    ctx.fillStyle = m.color || '#39d98a';
    ctx.beginPath(); ctx.arc(xx,yy,4,0,2*Math.PI); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.font = '12px system-ui';
    ctx.fillText(m.label, Math.min(xx+6, w-120), Math.max(yy-8, pad.t+12));
  });

  return {scaleX:X, scaleY:Y, bounds:{xmin,xmax,ymin,ymax}, pad};
}

// Hover tooltip on main chart
function attachHoverTooltip(canvas, scales, curveEval){
  const tt = chartTooltip;
  canvas.onmousemove = (e)=>{
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const invX = (x - scales.pad.l) / (canvas.width - scales.pad.l - scales.pad.r) * (scales.bounds.xmax - scales.bounds.xmin) + scales.bounds.xmin;
    if(invX < scales.bounds.xmin || invX > scales.bounds.xmax){ tt.hidden = true; return; }
    const val = curveEval(invX);
    const lines = [
      `Spend: ${fmt.usd0(invX)}`,
      `Ad Sales: ${fmt.usd0(val.y)}`,
      `ROAS: ${fmt.x2(val.y/Math.max(1e-9, invX))}x`,
      `mROAS: ${fmt.x2(val.df)} (Δ$ sales per $1 spend)`,
      `Elasticity: ${fmt.x2(val.elasticity)}`
    ];
    tt.innerHTML = lines.join('<br/>');
    tt.style.left = Math.min(rect.width - 220, x + 16) + 'px';
    tt.style.top = Math.max(6, y - 10) + 'px';
    tt.hidden = false;
  };
  canvas.onmouseleave = ()=>{ tt.hidden = true; };
}

// ---------- UI Update ----------
function renderResults(res){
  resultsGrid.innerHTML = '';

  function card(label, value, sub='', cls=''){
    const el = document.createElement('div');
    el.className = `metric ${cls}`;
    el.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>${sub?`<div class="sub">${sub}</div>`:''}`;
    resultsGrid.appendChild(el);
  }

  card('Model Selected', res.best.model, res.best.useTotal ? 'Joint fit: Ad + Total' : 'Ad-only fit');
  card('R² (Ad Sales)', fmt.x2(res.best.r2Ad));
  if(res.best.useTotal) card('R² (Total Sales)', fmt.x2(res.best.r2Tot||NaN));

  card('Current Spend', fmt.usd0(res.current), `ROAS ${fmt.x2(res.roasCur)}x · mROAS ${fmt.x2(res.mroasCur)} · Profit ${fmt.usd0(res.profitCur)}`);
  card('Optimal Spend (Profit Max)', fmt.usd0(res.xOpt), `ROAS ${fmt.x2(res.roasOpt)}x · mROAS ${fmt.x2(res.mroasOpt)} · Profit ${fmt.usd0(res.profitOpt)}`, 'ok');
  card('Knee (Diminishing Returns)', fmt.usd0(res.xKnee), 'Alt. visual elbow');

  const headroomClass = res.spendHeadroom>0 ? 'ok' : (Math.abs(res.spendHeadroom)<1 ? '' : 'bad');
  card('Spend Headroom', res.spendHeadroom>=0 ? fmt.usd0(res.spendHeadroom) : `▼ ${fmt.usd0(-res.spendHeadroom)}`, 'Optimal − Current', headroomClass);
  card('Ad Sales Headroom', res.adSalesHeadroom>=0 ? fmt.usd0(res.adSalesHeadroom) : `▼ ${fmt.usd0(-res.adSalesHeadroom)}`, 'At optimal vs current');
  card('ROAS Headroom', (res.roasHeadroom>=0?'+':'') + fmt.x2(res.roasHeadroom), 'At optimal vs current');
  card('Profit Headroom', res.profitHeadroom>=0 ? fmt.usd0(res.profitHeadroom) : `▼ ${fmt.usd0(-res.profitHeadroom)}`, 'At optimal vs current', headroomClass);

  card('Elasticity @ Current', fmt.x2(res.elasticityCur));
  card('Elasticity @ Optimal', fmt.x2(res.elasticityOpt));

  // Legend
  legendMain.innerHTML = `
    <div class="item"><div class="dot" style="background:#6ee7ff"></div> Historical (scatter)</div>
    <div class="item"><div class="dot" style="background:#9b8cff"></div> Fitted curve</div>
    <div class="item"><div class="dot" style="background:#39d98a"></div> Optimal</div>
    <div class="item"><div class="dot" style="background:#ffd166"></div> Knee</div>
    <div class="item"><div class="dot" style="background:#ff6b6b"></div> Current</div>
  `;
}

function renderCharts(res){
  // Main chart
  const mainCanvas = document.getElementById('chartMain');
  const points = res.xs.map((x,i)=> ({x, y: isFinite(res.yAds[i])?res.yAds[i]: (isFinite(res.yTotals[i])?res.yTotals[i]:0)}));
  const line = res.grid.x.map((x,i)=> ({x, y: res.grid.y[i]}));

  const markers = [
    {x: res.xOpt, y: (res.best.model==='Hill'? Model.hill.f(res.xOpt, res.best.params) : Model.expSat.f(res.xOpt, res.best.params)), label:'Optimal', color:'#39d98a'},
    {x: res.xKnee, y: (res.best.model==='Hill'? Model.hill.f(res.xKnee, res.best.params) : Model.expSat.f(res.xKnee, res.best.params)), label:'Knee', color:'#ffd166'},
    {x: res.current, y: (res.best.model==='Hill'? Model.hill.f(res.current, res.best.params) : Model.expSat.f(res.current, res.best.params)), label:'Current', color:'#ff6b6b'}
  ];

  const scales = plotScatterLine(mainCanvas, points, line, markers, {});
  attachHoverTooltip(mainCanvas, scales, (xx)=>{
    const y = (res.best.model==='Hill'? Model.hill.f(xx, res.best.params) : Model.expSat.f(xx, res.best.params));
    const df = (res.best.model==='Hill'? Model.hill.df(xx, res.best.params) : Model.expSat.df(xx, res.best.params));
    const elasticity = (y>0 && xx>0) ? (df*xx/y) : NaN;
    return {y, df, elasticity};
  });

  // mROAS chart
  const xg = res.grid.x;
  const yg = xg.map(x => (res.best.model==='Hill'? Model.hill.df(x, res.best.params) : Model.expSat.df(x, res.best.params)));
  const m1 = {x: res.xOpt, y: (res.best.model==='Hill'? Model.hill.df(res.xOpt, res.best.params) : Model.expSat.df(res.xOpt, res.best.params)), label:'Optimal', color:'#39d98a'};
  plotLine(document.getElementById('chartMarginal'), xg, yg, '$ Spend', 'mROAS', [m1], {yfmt:(v)=>v.toFixed(2)});

  // Profit chart
  const margin = parseFloat(marginInput.value)/100;
  const yp = xg.map(x => margin * (res.best.model==='Hill'? Model.hill.f(x, res.best.params) : Model.expSat.f(x, res.best.params)) - x);
  const m2 = {x: res.xOpt, y: margin*(res.best.model==='Hill'? Model.hill.f(res.xOpt, res.best.params) : Model.expSat.f(res.xOpt, res.best.params)) - res.xOpt, label:'Optimal', color:'#39d98a'};
  plotLine(document.getElementById('chartProfit'), xg, yp, '$ Spend', '$ Profit', [m2], {yfmt:(v)=>'$'+v.toFixed(0)});
}

// ---------- Export ----------
function exportResultsCSV(res){
  const lines = [];
  lines.push(['Metric','Value','Notes'].join(','));
  lines.push(['Model', res.best.model, res.best.useTotal?'Joint fit (Ad + Total)':'Ad-only fit'].join(','));
  lines.push(['R2_Ad', res.best.r2Ad.toFixed(4), '']);
  if(res.best.useTotal) lines.push(['R2_Total', (res.best.r2Tot||NaN).toFixed(4), '']);
  lines.push(['Current_Spend', res.current.toFixed(2), '']);
  lines.push(['Current_ROAS', res.roasCur.toFixed(4), '']);
  lines.push(['Current_mROAS', res.mroasCur.toFixed(6), '']);
  lines.push(['Current_Profit', res.profitCur.toFixed(2), '']);
  lines.push(['Optimal_Spend', res.xOpt.toFixed(2), 'margin * mROAS = 1']);
  lines.push(['Optimal_ROAS', res.roasOpt.toFixed(4), '']);
  lines.push(['Optimal_mROAS', res.mroasOpt.toFixed(6), '']);
  lines.push(['Optimal_Profit', res.profitOpt.toFixed(2), '']);
  lines.push(['Knee_Spend', res.xKnee.toFixed(2), 'Visual elbow']);
  lines.push(['Spend_Headroom', res.spendHeadroom.toFixed(2), 'Optimal − Current']);
  lines.push(['ROAS_Headroom', res.roasHeadroom.toFixed(4), 'Optimal − Current']);
  lines.push(['AdSales_Headroom', res.adSalesHeadroom.toFixed(2), 'Optimal − Current']);
  lines.push(['Profit_Headroom', res.profitHeadroom.toFixed(2), 'Optimal − Current']);
  lines.push(['Elasticity_Current', res.elasticityCur.toFixed(4), '']);
  lines.push(['Elasticity_Optimal', res.elasticityOpt.toFixed(4), '']);

  const blob = new Blob([lines.join('\n')], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'mroas_results.csv';
  a.click();
  setTimeout(()=>URL.revokeObjectURL(url), 500);
}

function exportChartPNG(){
  const canvas = document.getElementById('chartMain');
  const url = canvas.toDataURL("image/png");
  const a = document.createElement('a');
  a.href = url; a.download = 'mroas_chart.png';
  a.click();
}

// ---------- Gemini Summary ----------
async function generateSummary(res){
  const payload = {
    current: {
      spend: res.current, roas: res.roasCur, mroas: res.mroasCur, profit: res.profitCur
    },
    optimal: {
      spend: res.xOpt, roas: res.roasOpt, mroas: res.mroasOpt, profit: res.profitOpt
    },
    headroom: {
      spend: res.spendHeadroom, roas: res.roasHeadroom, adSales: res.adSalesHeadroom, profit: res.profitHeadroom
    },
    model: res.best.model, r2Ad: res.best.r2Ad, r2Tot: res.best.r2Tot || null,
    knee: res.xKnee, margin: parseFloat(marginInput.value)/100
  };

  const prompt = `You are a marketing analyst. Explain these results to a non-technical Amazon seller in 4–7 short bullet points. Avoid jargon. Be specific with dollar amounts and x-multipliers.
DATA: ${JSON.stringify(payload)}`;

  const key = geminiKey.value.trim();
  const model = geminiModel.value.trim();

  if(key){
    try{
      const body = {
        contents: [{role:"user", parts:[{text: prompt}]}],
        generationConfig: {temperature: 0.3}
      };
      let resp;
      if(useProxy.value==='yes'){
        resp = await fetch('/server/gemini', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({model, key, body})
        });
      }else{
        // Direct call (may hit CORS depending on your browser setup)
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(key)}`;
        resp = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      }
      const data = await resp.json();
      let text = "";
      try{
        text = data.candidates[0].content.parts.map(p=>p.text).join('\n');
      }catch(e){
        text = `Could not parse model response. Raw:\n${JSON.stringify(data, null, 2)}`;
      }
      summaryText.value = text;
    }catch(err){
      summaryText.value = "LLM request failed. Using built-in summary.\n\n" + fallbackSummary(res);
    }
  }else{
    summaryText.value = fallbackSummary(res);
  }
}

function fallbackSummary(res){
  const lines = [];
  lines.push(`• Optimal spend is ${fmt.usd0(res.xOpt)}, where the next $1 returns ~$${fmt.x2(res.mroasOpt)} in attributed sales and margin × mROAS ≈ 1.`);
  if(res.spendHeadroom>0){
    lines.push(`• There’s ~${fmt.usd0(res.spendHeadroom)} of scalable headroom before diminishing returns make the next dollar unprofitable.`);
  }else{
    lines.push(`• Current spend is above the profit point by ~${fmt.usd0(-res.spendHeadroom)}; each extra dollar likely erodes profit at your margin.`);
  }
  lines.push(`• At current spend (${fmt.usd0(res.current)}), ROAS ≈ ${fmt.x2(res.roasCur)}x; at optimal, ROAS ≈ ${fmt.x2(res.roasOpt)}x.`);
  lines.push(`• Expected incremental ad sales improvement: ${res.adSalesHeadroom>=0?'+':''}${fmt.usd0(res.adSalesHeadroom)} (optimal vs current).`);
  lines.push(`• Fit quality: R² (ad) = ${fmt.x2(res.best.r2Ad)}${res.best.useTotal?`, R² (total) = ${fmt.x2(res.best.r2Tot||NaN)}`:''}.`);
  lines.push(`• Elasticity now ≈ ${fmt.x2(res.elasticityCur)}, at optimal ≈ ${fmt.x2(res.elasticityOpt)} (slope flattens as spend rises).`);
  return lines.join('\n');
}

// ---------- Events ----------
fileInput.addEventListener('change', (e)=>{
  const f = e.target.files[0];
  if(!f) return;
  const reader = new FileReader();
  reader.onload = (ev)=>{
    rawCSV = ev.target.result;
    const parsed = parseCSV(rawCSV);
    headers = parsed.header;
    rows = parsed.rows;
    populateMappingSelectors();
    data = buildData();
    btnFit.disabled = data.length<5;
    btnGemini.disabled = true;
  };
  reader.readAsText(f);
});

[colSpend, colAdSales, colTotalSales, colROAS, colTROAS, colDate].forEach(sel => {
  sel.addEventListener('change', ()=>{
    mapping = {
      spend: colSpend.value || null,
      adSales: colAdSales.value || null,
      totalSales: colTotalSales.value || null,
      roas: colROAS.value || null,
      troas: colTROAS.value || null,
      date: colDate.value || null
    };
    data = buildData();
    btnFit.disabled = data.length<5;
  });
});

btnFit.addEventListener('click', ()=>{
  if(data.length<5) return;
  fitResult = fitModels(data);
  renderResults(fitResult);
  renderCharts(fitResult);
  btnExportCSV.disabled = false;
  btnExportPNG.disabled = false;
  btnGemini.disabled = false;
  // enable session export link
  const sess = {
    mapping, margin: parseFloat(marginInput.value), useTotal: useTotalSelect.value,
    currentMode: currentSpendMode.value, data: data
  };
  const blob = new Blob([JSON.stringify(sess)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  btnExportSession.href = url;
});

btnExportCSV.addEventListener('click', ()=>{
  if(!fitResult) return;
  exportResultsCSV(fitResult);
});
btnExportPNG.addEventListener('click', exportChartPNG);

btnGemini.addEventListener('click', ()=>{
  if(!fitResult) return;
  generateSummary(fitResult);
});

btnLoadSample.addEventListener('click', async ()=>{
  if(sampleLoaded) return;
  const resp = await fetch('SAMPLE_DATA.csv');
  const text = await resp.text();
  rawCSV = text;
  const parsed = parseCSV(rawCSV);
  headers = parsed.header;
  rows = parsed.rows;
  populateMappingSelectors();
  data = buildData();
  btnFit.disabled = data.length<5;
  sampleLoaded = true;
});

// Ready
document.addEventListener('DOMContentLoaded', ()=>{
  // nothing else
});
