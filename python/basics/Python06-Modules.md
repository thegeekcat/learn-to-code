- Modules:
 - Modules = Code libraries


```python
# Import Statement
import math 
```


```python
math.pi    
math.pow(2,10)  
math.log(100)
```




    4.605170185988092




```python
# List aa variable functions in a module
dir(math)
```




    ['__doc__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     'acos',
     'acosh',
     'asin',
     'asinh',
     'atan',
     'atan2',
     'atanh',
     'ceil',
     'comb',
     'copysign',
     'cos',
     'cosh',
     'degrees',
     'dist',
     'e',
     'erf',
     'erfc',
     'exp',
     'expm1',
     'fabs',
     'factorial',
     'floor',
     'fmod',
     'frexp',
     'fsum',
     'gamma',
     'gcd',
     'hypot',
     'inf',
     'isclose',
     'isfinite',
     'isinf',
     'isnan',
     'isqrt',
     'lcm',
     'ldexp',
     'lgamma',
     'log',
     'log10',
     'log1p',
     'log2',
     'modf',
     'nan',
     'nextafter',
     'perm',
     'pi',
     'pow',
     'prod',
     'radians',
     'remainder',
     'sin',
     'sinh',
     'sqrt',
     'tan',
     'tanh',
     'tau',
     'trunc',
     'ulp']




```python
# Get descriptions of a module
help(math)
```

    Help on built-in module math:
    
    NAME
        math
    
    DESCRIPTION
        This module provides access to the mathematical functions
        defined by the C standard.
    
    FUNCTIONS
        acos(x, /)
            Return the arc cosine (measured in radians) of x.
            
            The result is between 0 and pi.
        
        acosh(x, /)
            Return the inverse hyperbolic cosine of x.
        
        asin(x, /)
            Return the arc sine (measured in radians) of x.
            
            The result is between -pi/2 and pi/2.
        
        asinh(x, /)
            Return the inverse hyperbolic sine of x.
        
        atan(x, /)
            Return the arc tangent (measured in radians) of x.
            
            The result is between -pi/2 and pi/2.
        
        atan2(y, x, /)
            Return the arc tangent (measured in radians) of y/x.
            
            Unlike atan(y/x), the signs of both x and y are considered.
        
        atanh(x, /)
            Return the inverse hyperbolic tangent of x.
        
        ceil(x, /)
            Return the ceiling of x as an Integral.
            
            This is the smallest integer >= x.
        
        comb(n, k, /)
            Number of ways to choose k items from n items without repetition and without order.
            
            Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates
            to zero when k > n.
            
            Also called the binomial coefficient because it is equivalent
            to the coefficient of k-th term in polynomial expansion of the
            expression (1 + x)**n.
            
            Raises TypeError if either of the arguments are not integers.
            Raises ValueError if either of the arguments are negative.
        
        copysign(x, y, /)
            Return a float with the magnitude (absolute value) of x but the sign of y.
            
            On platforms that support signed zeros, copysign(1.0, -0.0)
            returns -1.0.
        
        cos(x, /)
            Return the cosine of x (measured in radians).
        
        cosh(x, /)
            Return the hyperbolic cosine of x.
        
        degrees(x, /)
            Convert angle x from radians to degrees.
        
        dist(p, q, /)
            Return the Euclidean distance between two points p and q.
            
            The points should be specified as sequences (or iterables) of
            coordinates.  Both inputs must have the same dimension.
            
            Roughly equivalent to:
                sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))
        
        erf(x, /)
            Error function at x.
        
        erfc(x, /)
            Complementary error function at x.
        
        exp(x, /)
            Return e raised to the power of x.
        
        expm1(x, /)
            Return exp(x)-1.
            
            This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x.
        
        fabs(x, /)
            Return the absolute value of the float x.
        
        factorial(x, /)
            Find x!.
            
            Raise a ValueError if x is negative or non-integral.
        
        floor(x, /)
            Return the floor of x as an Integral.
            
            This is the largest integer <= x.
        
        fmod(x, y, /)
            Return fmod(x, y), according to platform C.
            
            x % y may differ.
        
        frexp(x, /)
            Return the mantissa and exponent of x, as pair (m, e).
            
            m is a float and e is an int, such that x = m * 2.**e.
            If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0.
        
        fsum(seq, /)
            Return an accurate floating point sum of values in the iterable seq.
            
            Assumes IEEE-754 floating point arithmetic.
        
        gamma(x, /)
            Gamma function at x.
        
        gcd(*integers)
            Greatest Common Divisor.
        
        hypot(...)
            hypot(*coordinates) -> value
            
            Multidimensional Euclidean distance from the origin to a point.
            
            Roughly equivalent to:
                sqrt(sum(x**2 for x in coordinates))
            
            For a two dimensional point (x, y), gives the hypotenuse
            using the Pythagorean theorem:  sqrt(x*x + y*y).
            
            For example, the hypotenuse of a 3/4/5 right triangle is:
            
                >>> hypot(3.0, 4.0)
                5.0
        
        isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)
            Determine whether two floating point numbers are close in value.
            
              rel_tol
                maximum difference for being considered "close", relative to the
                magnitude of the input values
              abs_tol
                maximum difference for being considered "close", regardless of the
                magnitude of the input values
            
            Return True if a is close in value to b, and False otherwise.
            
            For the values to be considered close, the difference between them
            must be smaller than at least one of the tolerances.
            
            -inf, inf and NaN behave similarly to the IEEE 754 Standard.  That
            is, NaN is not close to anything, even itself.  inf and -inf are
            only close to themselves.
        
        isfinite(x, /)
            Return True if x is neither an infinity nor a NaN, and False otherwise.
        
        isinf(x, /)
            Return True if x is a positive or negative infinity, and False otherwise.
        
        isnan(x, /)
            Return True if x is a NaN (not a number), and False otherwise.
        
        isqrt(n, /)
            Return the integer part of the square root of the input.
        
        lcm(*integers)
            Least Common Multiple.
        
        ldexp(x, i, /)
            Return x * (2**i).
            
            This is essentially the inverse of frexp().
        
        lgamma(x, /)
            Natural logarithm of absolute value of Gamma function at x.
        
        log(...)
            log(x, [base=math.e])
            Return the logarithm of x to the given base.
            
            If the base not specified, returns the natural logarithm (base e) of x.
        
        log10(x, /)
            Return the base 10 logarithm of x.
        
        log1p(x, /)
            Return the natural logarithm of 1+x (base e).
            
            The result is computed in a way which is accurate for x near zero.
        
        log2(x, /)
            Return the base 2 logarithm of x.
        
        modf(x, /)
            Return the fractional and integer parts of x.
            
            Both results carry the sign of x and are floats.
        
        nextafter(x, y, /)
            Return the next floating-point value after x towards y.
        
        perm(n, k=None, /)
            Number of ways to choose k items from n items without repetition and with order.
            
            Evaluates to n! / (n - k)! when k <= n and evaluates
            to zero when k > n.
            
            If k is not specified or is None, then k defaults to n
            and the function returns n!.
            
            Raises TypeError if either of the arguments are not integers.
            Raises ValueError if either of the arguments are negative.
        
        pow(x, y, /)
            Return x**y (x to the power of y).
        
        prod(iterable, /, *, start=1)
            Calculate the product of all the elements in the input iterable.
            
            The default start value for the product is 1.
            
            When the iterable is empty, return the start value.  This function is
            intended specifically for use with numeric values and may reject
            non-numeric types.
        
        radians(x, /)
            Convert angle x from degrees to radians.
        
        remainder(x, y, /)
            Difference between x and the closest integer multiple of y.
            
            Return x - n*y where n*y is the closest integer multiple of y.
            In the case where x is exactly halfway between two multiples of
            y, the nearest even value of n is used. The result is always exact.
        
        sin(x, /)
            Return the sine of x (measured in radians).
        
        sinh(x, /)
            Return the hyperbolic sine of x.
        
        sqrt(x, /)
            Return the square root of x.
        
        tan(x, /)
            Return the tangent of x (measured in radians).
        
        tanh(x, /)
            Return the hyperbolic tangent of x.
        
        trunc(x, /)
            Truncates the Real x to the nearest Integral toward 0.
            
            Uses the __trunc__ magic method.
        
        ulp(x, /)
            Return the value of the least significant bit of the float x.
    
    DATA
        e = 2.718281828459045
        inf = inf
        nan = nan
        pi = 3.141592653589793
        tau = 6.283185307179586
    
    FILE
        (built-in)
    
    
    


```python
from ftplib import FTP
```


```python
ftp = FTP('ftp1.at.proftpd.org')
```


```python
ftp.login()
```




    '230 Anonymous access granted, restrictions apply'




```python
ftp.retrlines('LIST')   # A list of the ftp server
```

    -rw-rw-r--   1 ftp      ftp           451 Jul  1  2005 README.MIRRORS
    drwxrwxr-x   3 ftp      ftp          4096 Jul  1  2005 devel
    drwxrwxr-x   3 ftp      ftp          4096 Dec  2  2010 distrib
    drwxrwxr-x   4 ftp      ftp          4096 Jul  1  2005 historic
    




    '226 Transfer complete'




```python
ftp.quit()  # Quit the connection
```




    '221 Goodbye.'




```python
import requests
```


```python
help(requests)
```

    Help on package requests:
    
    NAME
        requests
    
    DESCRIPTION
        Requests HTTP Library
        ~~~~~~~~~~~~~~~~~~~~~
        
        Requests is an HTTP library, written in Python, for human beings.
        Basic GET usage:
        
           >>> import requests
           >>> r = requests.get('https://www.python.org')
           >>> r.status_code
           200
           >>> b'Python is a programming language' in r.content
           True
        
        ... or POST:
        
           >>> payload = dict(key1='value1', key2='value2')
           >>> r = requests.post('https://httpbin.org/post', data=payload)
           >>> print(r.text)
           {
             ...
             "form": {
               "key1": "value1",
               "key2": "value2"
             },
             ...
           }
        
        The other HTTP methods are supported - see `requests.api`. Full documentation
        is at <https://requests.readthedocs.io>.
        
        :copyright: (c) 2017 by Kenneth Reitz.
        :license: Apache 2.0, see LICENSE for more details.
    
    PACKAGE CONTENTS
        __version__
        _internal_utils
        adapters
        api
        auth
        certs
        compat
        cookies
        exceptions
        help
        hooks
        models
        packages
        sessions
        status_codes
        structures
        utils
    
    FUNCTIONS
        check_compatibility(urllib3_version, chardet_version, charset_normalizer_version)
    
    DATA
        __author_email__ = 'me@kennethreitz.org'
        __build__ = 141057
        __cake__ = '✨ 🍰 ✨'
        __copyright__ = 'Copyright 2022 Kenneth Reitz'
        __description__ = 'Python HTTP for Humans.'
        __license__ = 'Apache 2.0'
        __title__ = 'requests'
        __url__ = 'https://requests.readthedocs.io'
        chardet_version = '4.0.0'
        charset_normalizer_version = '2.0.12'
        codes = <lookup 'status_codes'>
    
    VERSION
        2.27.1
    
    AUTHOR
        Kenneth Reitz
    
    FILE
        /usr/local/lib/python3.9/dist-packages/requests/__init__.py
    
    
    


```python
response = requests.get('http://www.google.com')
```


```python
response.text  # Web-crawling
```




    '<!doctype html><html itemscope="" itemtype="http://schema.org/WebPage" lang="en"><head><meta content="Search the world\'s information, including webpages, images, videos and more. Google has many special features to help you find exactly what you\'re looking for." name="description"><meta content="noodp" name="robots"><meta content="text/html; charset=UTF-8" http-equiv="Content-Type"><meta content="/images/branding/googleg/1x/googleg_standard_color_128dp.png" itemprop="image"><title>Google</title><script nonce="JnPzkVwMwqmad8kFEdQDRw">(function(){window.google={kEI:\'XpZGZIWDM9miqtsPvp-oqAw\',kEXPI:\'0,1303456,55953,6058,207,4804,2316,383,246,5,1129120,1197694,707,380089,16115,19397,9287,22431,1361,12316,2819,14764,4998,13228,3847,6884,31560,885,1987,2891,561,3365,214,4208,3406,606,30668,30022,6398,8926,432,3,1590,1,16916,2652,4,1528,2304,29062,13065,13658,2980,1457,16786,5815,2542,4094,7596,1,39042,1,3111,2,14022,2373,342,23024,5679,1021,25046,6075,4568,6259,23418,1252,5835,14968,4332,7484,445,2,2,1,26632,8155,6680,701,2,3,1475,14489,873,9626,10008,7,1922,9779,22887,13267,2524,3781,20199,927,19209,14,82,16514,3692,109,2412,5856,3785,15175,4415,988,2265,765,6110,3226,6480,1804,7734,495,1150,1093,2885,151,298,5969,6057,6850,415,4585,1195,3048,1195,935,43,2322,649,13,1635,1606,1635,1,2562,2,2142,891,868,2708,2571,6103,1666,865,1112,698,1268,66,1,1926,176,3975,344,173,119,343,4414,618,330,379,2,296,730,915,123,46,1018,1,729,765,55,171,772,90,107,514,419,849,204,3,3,9,1075,718,101,2,1477,37,7,119,40,93,114,325,5,868,34,104,209,386,272,40,259,498,291,4,192,151,121,255,253,179,591,186,2,10,44,614,1820,49,1,173,5206381,91,190,2,70,8798843,3311,141,795,19735,1,1,348,5005,31,19,5,16,17,31,5,3,6,14,3,2,1,135,13,12,50,23646002,299113,396,4041746,1964,1007,15665,2894,6250,15739,180,1146,388,1413343,146987,23612606,444,93,31,102,554,419,1103,1,235,1,761,76,2,332,16,178,361,351,2,804,3984,292,389,4,457\',kBL:\'WWhe\',kOPI:89978449};google.sn=\'webhp\';google.kHL=\'en\';})();(function(){\nvar e=this||self;var g,h=[];function k(a){for(var c;a&&(!a.getAttribute||!(c=a.getAttribute("eid")));)a=a.parentNode;return c||g}function l(a){for(var c=null;a&&(!a.getAttribute||!(c=a.getAttribute("leid")));)a=a.parentNode;return c}function m(a){/^http:/i.test(a)&&"https:"===window.location.protocol&&(google.ml&&google.ml(Error("a"),!1,{src:a,glmm:1}),a="");return a}\nfunction p(a,c,b,f){var d="";-1===c.search("&ei=")&&(d="&ei="+k(b),-1===c.search("&lei=")&&(b=l(b))&&(d+="&lei="+b));b="";e._cshid&&-1===c.search("&cshid=")&&"slh"!==a&&(b="&cshid="+e._cshid);return"/"+(f||"gen_204")+"?atyp=i&ct="+String(a)+"&cad="+(c+d)+"&zx="+String(Date.now())+b};g=google.kEI;google.getEI=k;google.getLEI=l;google.ml=function(){return null};google.log=function(a,c,b,f,d){b||(b=p(a,c,f,d));if(b=m(b)){a=new Image;var n=h.length;h[n]=a;a.onerror=a.onload=a.onabort=function(){delete h[n]};a.src=b}};google.logUrl=function(a){return p("",a)};}).call(this);(function(){google.y={};google.sy=[];google.x=function(a,b){if(a)var c=a.id;else{do c=Math.random();while(google.y[c])}google.y[c]=[a,b];return!1};google.sx=function(a){google.sy.push(a)};google.lm=[];google.plm=function(a){google.lm.push.apply(google.lm,a)};google.lq=[];google.load=function(a,b,c){google.lq.push([[a],b,c])};google.loadAll=function(a,b){google.lq.push([a,b])};google.bx=!1;google.lx=function(){};}).call(this);google.f={};(function(){\ndocument.documentElement.addEventListener("submit",function(b){var a;if(a=b.target){var c=a.getAttribute("data-submitfalse");a="1"===c||"q"===c&&!a.elements.q.value?!0:!1}else a=!1;a&&(b.preventDefault(),b.stopPropagation())},!0);document.documentElement.addEventListener("click",function(b){var a;a:{for(a=b.target;a&&a!==document.documentElement;a=a.parentElement)if("A"===a.tagName){a="1"===a.getAttribute("data-nohref");break a}a=!1}a&&b.preventDefault()},!0);}).call(this);</script><style>#gbar,#guser{font-size:13px;padding-top:1px !important;}#gbar{height:22px}#guser{padding-bottom:7px !important;text-align:right}.gbh,.gbd{border-top:1px solid #c9d7f1;font-size:1px}.gbh{height:0;position:absolute;top:24px;width:100%}@media all{.gb1{height:22px;margin-right:.5em;vertical-align:top}#gbar{float:left}}a.gb1,a.gb4{text-decoration:underline !important}a.gb1,a.gb4{color:#00c !important}.gbi .gb4{color:#dd8e27 !important}.gbf .gb4{color:#900 !important}\n</style><style>body,td,a,p,.h{font-family:arial,sans-serif}body{margin:0;overflow-y:scroll}#gog{padding:3px 8px 0}td{line-height:.8em}.gac_m td{line-height:17px}form{margin-bottom:20px}.h{color:#1558d6}em{font-weight:bold;font-style:normal}.lst{height:25px;width:496px}.gsfi,.lst{font:18px arial,sans-serif}.gsfs{font:17px arial,sans-serif}.ds{display:inline-box;display:inline-block;margin:3px 0 4px;margin-left:4px}input{font-family:inherit}body{background:#fff;color:#000}a{color:#4b11a8;text-decoration:none}a:hover,a:active{text-decoration:underline}.fl a{color:#1558d6}a:visited{color:#4b11a8}.sblc{padding-top:5px}.sblc a{display:block;margin:2px 0;margin-left:13px;font-size:11px}.lsbb{background:#f8f9fa;border:solid 1px;border-color:#dadce0 #70757a #70757a #dadce0;height:30px}.lsbb{display:block}#WqQANb a{display:inline-block;margin:0 12px}.lsb{background:url(/images/nav_logo229.png) 0 -261px repeat-x;border:none;color:#000;cursor:pointer;height:30px;margin:0;outline:0;font:15px arial,sans-serif;vertical-align:top}.lsb:active{background:#dadce0}.lst:focus{outline:none}</style><script nonce="JnPzkVwMwqmad8kFEdQDRw">(function(){window.google.erd={jsr:1,bv:1781,de:true};\nvar h=this||self;var k,l=null!=(k=h.mei)?k:1,n,p=null!=(n=h.sdo)?n:!0,q=0,r,t=google.erd,v=t.jsr;google.ml=function(a,b,d,m,e){e=void 0===e?2:e;b&&(r=a&&a.message);if(google.dl)return google.dl(a,e,d),null;if(0>v){window.console&&console.error(a,d);if(-2===v)throw a;b=!1}else b=!a||!a.message||"Error loading script"===a.message||q>=l&&!m?!1:!0;if(!b)return null;q++;d=d||{};b=encodeURIComponent;var c="/gen_204?atyp=i&ei="+b(google.kEI);google.kEXPI&&(c+="&jexpid="+b(google.kEXPI));c+="&srcpg="+b(google.sn)+"&jsr="+b(t.jsr)+"&bver="+b(t.bv);var f=a.lineNumber;void 0!==f&&(c+="&line="+f);var g=\na.fileName;g&&(0<g.indexOf("-extension:/")&&(e=3),c+="&script="+b(g),f&&g===window.location.href&&(f=document.documentElement.outerHTML.split("\\n")[f],c+="&cad="+b(f?f.substring(0,300):"No script found.")));c+="&jsel="+e;for(var u in d)c+="&",c+=b(u),c+="=",c+=b(d[u]);c=c+"&emsg="+b(a.name+": "+a.message);c=c+"&jsst="+b(a.stack||"N/A");12288<=c.length&&(c=c.substr(0,12288));a=c;m||google.log(0,"",a);return a};window.onerror=function(a,b,d,m,e){r!==a&&(a=e instanceof Error?e:Error(a),void 0===d||"lineNumber"in a||(a.lineNumber=d),void 0===b||"fileName"in a||(a.fileName=b),google.ml(a,!1,void 0,!1,"SyntaxError"===a.name||"SyntaxError"===a.message.substring(0,11)||-1!==a.message.indexOf("Script error")?3:0));r=null;p&&q>=l&&(window.onerror=null)};})();</script></head><body bgcolor="#fff"><script nonce="JnPzkVwMwqmad8kFEdQDRw">(function(){var src=\'/images/nav_logo229.png\';var iesg=false;document.body.onload = function(){window.n && window.n();if (document.images){new Image().src=src;}\nif (!iesg){document.f&&document.f.q.focus();document.gbqf&&document.gbqf.q.focus();}\n}\n})();</script><div id="mngb"><div id=gbar><nobr><b class=gb1>Search</b> <a class=gb1 href="http://www.google.com/imghp?hl=en&tab=wi">Images</a> <a class=gb1 href="http://maps.google.com/maps?hl=en&tab=wl">Maps</a> <a class=gb1 href="https://play.google.com/?hl=en&tab=w8">Play</a> <a class=gb1 href="https://www.youtube.com/?tab=w1">YouTube</a> <a class=gb1 href="https://news.google.com/?tab=wn">News</a> <a class=gb1 href="https://mail.google.com/mail/?tab=wm">Gmail</a> <a class=gb1 href="https://drive.google.com/?tab=wo">Drive</a> <a class=gb1 style="text-decoration:none" href="https://www.google.com/intl/en/about/products?tab=wh"><u>More</u> &raquo;</a></nobr></div><div id=guser width=100%><nobr><span id=gbn class=gbi></span><span id=gbf class=gbf></span><span id=gbe></span><a href="http://www.google.com/history/optout?hl=en" class=gb4>Web History</a> | <a  href="/preferences?hl=en" class=gb4>Settings</a> | <a target=_top id=gb_70 href="https://accounts.google.com/ServiceLogin?hl=en&passive=true&continue=http://www.google.com/&ec=GAZAAQ" class=gb4>Sign in</a></nobr></div><div class=gbh style=left:0></div><div class=gbh style=right:0></div></div><center><br clear="all" id="lgpd"><div id="lga"><img alt="Google" height="92" src="/images/branding/googlelogo/1x/googlelogo_white_background_color_272x92dp.png" style="padding:28px 0 14px" width="272" id="hplogo"><br><br></div><form action="/search" name="f"><table cellpadding="0" cellspacing="0"><tr valign="top"><td width="25%">&nbsp;</td><td align="center" nowrap=""><input name="ie" value="ISO-8859-1" type="hidden"><input value="en" name="hl" type="hidden"><input name="source" type="hidden" value="hp"><input name="biw" type="hidden"><input name="bih" type="hidden"><div class="ds" style="height:32px;margin:4px 0"><input class="lst" style="margin:0;padding:5px 8px 0 6px;vertical-align:top;color:#000" autocomplete="off" value="" title="Google Search" maxlength="2048" name="q" size="57"></div><br style="line-height:0"><span class="ds"><span class="lsbb"><input class="lsb" value="Google Search" name="btnG" type="submit"></span></span><span class="ds"><span class="lsbb"><input class="lsb" id="tsuid_1" value="I\'m Feeling Lucky" name="btnI" type="submit"><script nonce="JnPzkVwMwqmad8kFEdQDRw">(function(){var id=\'tsuid_1\';document.getElementById(id).onclick = function(){if (this.form.q.value){this.checked = 1;if (this.form.iflsig)this.form.iflsig.disabled = false;}\nelse top.location=\'/doodles/\';};})();</script><input value="AOEireoAAAAAZEakbp8v5WltqniBo3hyq6yU16A5PUQi" name="iflsig" type="hidden"></span></span></td><td class="fl sblc" align="left" nowrap="" width="25%"><a href="/advanced_search?hl=en&amp;authuser=0">Advanced search</a></td></tr></table><input id="gbv" name="gbv" type="hidden" value="1"><script nonce="JnPzkVwMwqmad8kFEdQDRw">(function(){var a,b="1";if(document&&document.getElementById)if("undefined"!=typeof XMLHttpRequest)b="2";else if("undefined"!=typeof ActiveXObject){var c,d,e=["MSXML2.XMLHTTP.6.0","MSXML2.XMLHTTP.3.0","MSXML2.XMLHTTP","Microsoft.XMLHTTP"];for(c=0;d=e[c++];)try{new ActiveXObject(d),b="2"}catch(h){}}a=b;if("2"==a&&-1==location.search.indexOf("&gbv=2")){var f=google.gbvu,g=document.getElementById("gbv");g&&(g.value=a);f&&window.setTimeout(function(){location.href=f},0)};}).call(this);</script></form><div id="gac_scont"></div><div style="font-size:83%;min-height:3.5em"><br><div id="prm"><style>.szppmdbYutt__middle-slot-promo{font-size:small;margin-bottom:32px}.szppmdbYutt__middle-slot-promo a.ZIeIlb{display:inline-block;text-decoration:none}.szppmdbYutt__middle-slot-promo img{border:none;margin-right:5px;vertical-align:middle}</style><div class="szppmdbYutt__middle-slot-promo" data-ved="0ahUKEwiFquKf4cL-AhVZkWoFHb4PCsUQnIcBCAQ"><a class="NKcBbd" href="https://www.google.com/url?q=https://artsandculture.google.com/experiment/zgFx1tMqeIZyTw%3Futm_source%3Dgoogle%26utm_medium%3Dhppromo%26utm_campaign%3Dcallinginourcorals&amp;source=hpp&amp;id=19034922&amp;ct=3&amp;usg=AOvVaw0nMWsnMoeASDuSYrKnPMNj&amp;sa=X&amp;ved=0ahUKEwiFquKf4cL-AhVZkWoFHb4PCsUQ8IcBCAU" rel="nofollow">Learn how to help restore coral reefs, simply by listening</a></div></div></div><span id="footer"><div style="font-size:10pt"><div style="margin:19px auto;text-align:center" id="WqQANb"><a href="/intl/en/ads/">Advertising</a><a href="/services/">Business Solutions</a><a href="/intl/en/about.html">About Google</a></div></div><p style="font-size:8pt;color:#70757a">&copy; 2023 - <a href="/intl/en/policies/privacy/">Privacy</a> - <a href="/intl/en/policies/terms/">Terms</a></p></span></center><script nonce="JnPzkVwMwqmad8kFEdQDRw">(function(){window.google.cdo={height:757,width:1440};(function(){var a=window.innerWidth,b=window.innerHeight;if(!a||!b){var c=window.document,d="CSS1Compat"==c.compatMode?c.documentElement:c.body;a=d.clientWidth;b=d.clientHeight}a&&b&&(a!=google.cdo.width||b!=google.cdo.height)&&google.log("","","/client_204?&atyp=i&biw="+a+"&bih="+b+"&ei="+google.kEI);}).call(this);})();</script> <script nonce="JnPzkVwMwqmad8kFEdQDRw">(function(){google.xjs={ck:\'xjs.hp.cZMjK1rN2dw.L.X.O\',cs:\'ACT90oGdWgvp7b-1i002ub8NTEiqmwqPag\',excm:[]};})();</script>  <script nonce="JnPzkVwMwqmad8kFEdQDRw">(function(){var u=\'/xjs/_/js/k\\x3dxjs.hp.en.qkDX73W2TvU.O/am\\x3dAAAAOgEAFABY/d\\x3d1/ed\\x3d1/rs\\x3dACT90oFpp8_uyj9hwoAl3W3tvYwd1PFWOg/m\\x3dsb_he,d\';var amd=0;\nvar e=this||self,f=function(c){return c};var h;var n=function(c,g){this.g=g===l?c:""};n.prototype.toString=function(){return this.g+""};var l={};\nfunction p(){var c=u,g=function(){};google.lx=google.stvsc?g:function(){google.timers&&google.timers.load&&google.tick&&google.tick("load","xjsls");var a=document;var b="SCRIPT";"application/xhtml+xml"===a.contentType&&(b=b.toLowerCase());b=a.createElement(b);a=null===c?"null":void 0===c?"undefined":c;if(void 0===h){var d=null;var m=e.trustedTypes;if(m&&m.createPolicy){try{d=m.createPolicy("goog#html",{createHTML:f,createScript:f,createScriptURL:f})}catch(r){e.console&&e.console.error(r.message)}h=\nd}else h=d}a=(d=h)?d.createScriptURL(a):a;a=new n(a,l);b.src=a instanceof n&&a.constructor===n?a.g:"type_error:TrustedResourceUrl";var k,q;(k=(a=null==(q=(k=(b.ownerDocument&&b.ownerDocument.defaultView||window).document).querySelector)?void 0:q.call(k,"script[nonce]"))?a.nonce||a.getAttribute("nonce")||"":"")&&b.setAttribute("nonce",k);document.body.appendChild(b);google.psa=!0;google.lx=g};google.bx||google.lx()};google.xjsu=u;e._F_jsUrl=u;setTimeout(function(){0<amd?google.caft(function(){return p()},amd):p()},0);})();window._ = window._ || {};window._DumpException = _._DumpException = function(e){throw e;};window._s = window._s || {};_s._DumpException = _._DumpException;window._qs = window._qs || {};_qs._DumpException = _._DumpException;function _F_installCss(c){}\n(function(){google.jl={blt:\'none\',chnk:0,dw:false,dwu:true,emtn:0,end:0,ico:false,ikb:0,ine:false,injs:\'none\',injt:0,injth:0,injv2:false,lls:\'default\',pdt:0,rep:0,snet:true,strt:0,ubm:false,uwp:true};})();(function(){var pmc=\'{\\x22d\\x22:{},\\x22sb_he\\x22:{\\x22agen\\x22:true,\\x22cgen\\x22:true,\\x22client\\x22:\\x22heirloom-hp\\x22,\\x22dh\\x22:true,\\x22ds\\x22:\\x22\\x22,\\x22fl\\x22:true,\\x22host\\x22:\\x22google.com\\x22,\\x22jsonp\\x22:true,\\x22msgs\\x22:{\\x22cibl\\x22:\\x22Clear Search\\x22,\\x22dym\\x22:\\x22Did you mean:\\x22,\\x22lcky\\x22:\\x22I\\\\u0026#39;m Feeling Lucky\\x22,\\x22lml\\x22:\\x22Learn more\\x22,\\x22psrc\\x22:\\x22This search was removed from your \\\\u003Ca href\\x3d\\\\\\x22/history\\\\\\x22\\\\u003EWeb History\\\\u003C/a\\\\u003E\\x22,\\x22psrl\\x22:\\x22Remove\\x22,\\x22sbit\\x22:\\x22Search by image\\x22,\\x22srch\\x22:\\x22Google Search\\x22},\\x22ovr\\x22:{},\\x22pq\\x22:\\x22\\x22,\\x22rfs\\x22:[],\\x22sbas\\x22:\\x220 3px 8px 0 rgba(0,0,0,0.2),0 0 0 1px rgba(0,0,0,0.08)\\x22,\\x22stok\\x22:\\x22dBYZ-UQAjc8jZU_KLMxjYRGHeOM\\x22}}\';google.pmc=JSON.parse(pmc);})();</script>       </body></html>'




```python
requests.get('http://www.google.com').text
```




    '<!doctype html><html itemscope="" itemtype="http://schema.org/WebPage" lang="en"><head><meta content="Search the world\'s information, including webpages, images, videos and more. Google has many special features to help you find exactly what you\'re looking for." name="description"><meta content="noodp" name="robots"><meta content="text/html; charset=UTF-8" http-equiv="Content-Type"><meta content="/images/branding/googleg/1x/googleg_standard_color_128dp.png" itemprop="image"><title>Google</title><script nonce="_ABgeM1p2MILLXMtY--z4A">(function(){window.google={kEI:\'ZZZGZOP4CrKrqtsP_PykoAw\',kEXPI:\'0,1303424,55985,1710,4348,207,4804,2316,383,246,5,1129120,1633,1196155,380703,16114,28684,22430,1362,283,12031,17585,4998,13226,3849,38444,2872,2891,3926,213,4210,3405,606,29877,791,30022,15324,432,3,1590,1,16916,2652,4,1528,2304,923,28139,13064,16639,1457,16786,5797,2560,4094,7596,1,42154,2,14022,2373,342,23024,5679,1021,31121,4569,6255,23421,1252,5835,14968,4332,7484,445,2,2,1,6960,17666,2006,8155,7381,2,3,15964,873,7830,1796,10008,7,1922,9779,36154,6305,20199,20136,14,82,2932,13582,3692,109,2412,850,699,4307,3785,4266,15321,991,1500,1530,1273,4837,2461,7245,1291,513,6249,1485,29,466,1150,1093,2885,449,2262,4446,5318,576,6273,416,2171,3609,3049,2129,1330,1697,477,773,385,1607,4197,2,2142,311,174,406,868,2708,1443,1128,2469,5,4868,427,1517,4,421,35,1966,1365,825,1168,277,754,1755,263,82,82,90,119,344,1237,997,2797,266,64,375,2,301,728,916,123,46,1018,1,1501,10,208,505,465,616,5,181,753,229,2851,739,36,127,135,427,4,879,32,61,45,209,386,90,1,315,165,459,32,87,28,1,134,245,154,116,256,253,181,589,186,2,10,44,611,1822,224,5206662,2,70,134,64,259,5993963,1203,2803220,3311,141,795,19735,1,347,5008,21,9,20,23,10,2,41,6,1,12,4,2,48,88,13,1,23945176,396,4041746,1964,14298,2375,2893,6250,15739,180,1146,884,1559833,34358,23578248,361,178,32,100,123,432,330,687,1,13,425,65,1,205,51,741,79,2,88,256,1697,702,436,1025,2459,504,185,523\',kBL:\'WWhe\',kOPI:89978449};google.sn=\'webhp\';google.kHL=\'en\';})();(function(){\nvar e=this||self;var g,h=[];function k(a){for(var c;a&&(!a.getAttribute||!(c=a.getAttribute("eid")));)a=a.parentNode;return c||g}function l(a){for(var c=null;a&&(!a.getAttribute||!(c=a.getAttribute("leid")));)a=a.parentNode;return c}function m(a){/^http:/i.test(a)&&"https:"===window.location.protocol&&(google.ml&&google.ml(Error("a"),!1,{src:a,glmm:1}),a="");return a}\nfunction p(a,c,b,f){var d="";-1===c.search("&ei=")&&(d="&ei="+k(b),-1===c.search("&lei=")&&(b=l(b))&&(d+="&lei="+b));b="";e._cshid&&-1===c.search("&cshid=")&&"slh"!==a&&(b="&cshid="+e._cshid);return"/"+(f||"gen_204")+"?atyp=i&ct="+String(a)+"&cad="+(c+d)+"&zx="+String(Date.now())+b};g=google.kEI;google.getEI=k;google.getLEI=l;google.ml=function(){return null};google.log=function(a,c,b,f,d){b||(b=p(a,c,f,d));if(b=m(b)){a=new Image;var n=h.length;h[n]=a;a.onerror=a.onload=a.onabort=function(){delete h[n]};a.src=b}};google.logUrl=function(a){return p("",a)};}).call(this);(function(){google.y={};google.sy=[];google.x=function(a,b){if(a)var c=a.id;else{do c=Math.random();while(google.y[c])}google.y[c]=[a,b];return!1};google.sx=function(a){google.sy.push(a)};google.lm=[];google.plm=function(a){google.lm.push.apply(google.lm,a)};google.lq=[];google.load=function(a,b,c){google.lq.push([[a],b,c])};google.loadAll=function(a,b){google.lq.push([a,b])};google.bx=!1;google.lx=function(){};}).call(this);google.f={};(function(){\ndocument.documentElement.addEventListener("submit",function(b){var a;if(a=b.target){var c=a.getAttribute("data-submitfalse");a="1"===c||"q"===c&&!a.elements.q.value?!0:!1}else a=!1;a&&(b.preventDefault(),b.stopPropagation())},!0);document.documentElement.addEventListener("click",function(b){var a;a:{for(a=b.target;a&&a!==document.documentElement;a=a.parentElement)if("A"===a.tagName){a="1"===a.getAttribute("data-nohref");break a}a=!1}a&&b.preventDefault()},!0);}).call(this);</script><style>#gbar,#guser{font-size:13px;padding-top:1px !important;}#gbar{height:22px}#guser{padding-bottom:7px !important;text-align:right}.gbh,.gbd{border-top:1px solid #c9d7f1;font-size:1px}.gbh{height:0;position:absolute;top:24px;width:100%}@media all{.gb1{height:22px;margin-right:.5em;vertical-align:top}#gbar{float:left}}a.gb1,a.gb4{text-decoration:underline !important}a.gb1,a.gb4{color:#00c !important}.gbi .gb4{color:#dd8e27 !important}.gbf .gb4{color:#900 !important}\n</style><style>body,td,a,p,.h{font-family:arial,sans-serif}body{margin:0;overflow-y:scroll}#gog{padding:3px 8px 0}td{line-height:.8em}.gac_m td{line-height:17px}form{margin-bottom:20px}.h{color:#1558d6}em{font-weight:bold;font-style:normal}.lst{height:25px;width:496px}.gsfi,.lst{font:18px arial,sans-serif}.gsfs{font:17px arial,sans-serif}.ds{display:inline-box;display:inline-block;margin:3px 0 4px;margin-left:4px}input{font-family:inherit}body{background:#fff;color:#000}a{color:#4b11a8;text-decoration:none}a:hover,a:active{text-decoration:underline}.fl a{color:#1558d6}a:visited{color:#4b11a8}.sblc{padding-top:5px}.sblc a{display:block;margin:2px 0;margin-left:13px;font-size:11px}.lsbb{background:#f8f9fa;border:solid 1px;border-color:#dadce0 #70757a #70757a #dadce0;height:30px}.lsbb{display:block}#WqQANb a{display:inline-block;margin:0 12px}.lsb{background:url(/images/nav_logo229.png) 0 -261px repeat-x;border:none;color:#000;cursor:pointer;height:30px;margin:0;outline:0;font:15px arial,sans-serif;vertical-align:top}.lsb:active{background:#dadce0}.lst:focus{outline:none}</style><script nonce="_ABgeM1p2MILLXMtY--z4A">(function(){window.google.erd={jsr:1,bv:1781,de:true};\nvar h=this||self;var k,l=null!=(k=h.mei)?k:1,n,p=null!=(n=h.sdo)?n:!0,q=0,r,t=google.erd,v=t.jsr;google.ml=function(a,b,d,m,e){e=void 0===e?2:e;b&&(r=a&&a.message);if(google.dl)return google.dl(a,e,d),null;if(0>v){window.console&&console.error(a,d);if(-2===v)throw a;b=!1}else b=!a||!a.message||"Error loading script"===a.message||q>=l&&!m?!1:!0;if(!b)return null;q++;d=d||{};b=encodeURIComponent;var c="/gen_204?atyp=i&ei="+b(google.kEI);google.kEXPI&&(c+="&jexpid="+b(google.kEXPI));c+="&srcpg="+b(google.sn)+"&jsr="+b(t.jsr)+"&bver="+b(t.bv);var f=a.lineNumber;void 0!==f&&(c+="&line="+f);var g=\na.fileName;g&&(0<g.indexOf("-extension:/")&&(e=3),c+="&script="+b(g),f&&g===window.location.href&&(f=document.documentElement.outerHTML.split("\\n")[f],c+="&cad="+b(f?f.substring(0,300):"No script found.")));c+="&jsel="+e;for(var u in d)c+="&",c+=b(u),c+="=",c+=b(d[u]);c=c+"&emsg="+b(a.name+": "+a.message);c=c+"&jsst="+b(a.stack||"N/A");12288<=c.length&&(c=c.substr(0,12288));a=c;m||google.log(0,"",a);return a};window.onerror=function(a,b,d,m,e){r!==a&&(a=e instanceof Error?e:Error(a),void 0===d||"lineNumber"in a||(a.lineNumber=d),void 0===b||"fileName"in a||(a.fileName=b),google.ml(a,!1,void 0,!1,"SyntaxError"===a.name||"SyntaxError"===a.message.substring(0,11)||-1!==a.message.indexOf("Script error")?3:0));r=null;p&&q>=l&&(window.onerror=null)};})();</script></head><body bgcolor="#fff"><script nonce="_ABgeM1p2MILLXMtY--z4A">(function(){var src=\'/images/nav_logo229.png\';var iesg=false;document.body.onload = function(){window.n && window.n();if (document.images){new Image().src=src;}\nif (!iesg){document.f&&document.f.q.focus();document.gbqf&&document.gbqf.q.focus();}\n}\n})();</script><div id="mngb"><div id=gbar><nobr><b class=gb1>Search</b> <a class=gb1 href="http://www.google.com/imghp?hl=en&tab=wi">Images</a> <a class=gb1 href="http://maps.google.com/maps?hl=en&tab=wl">Maps</a> <a class=gb1 href="https://play.google.com/?hl=en&tab=w8">Play</a> <a class=gb1 href="https://www.youtube.com/?tab=w1">YouTube</a> <a class=gb1 href="https://news.google.com/?tab=wn">News</a> <a class=gb1 href="https://mail.google.com/mail/?tab=wm">Gmail</a> <a class=gb1 href="https://drive.google.com/?tab=wo">Drive</a> <a class=gb1 style="text-decoration:none" href="https://www.google.com/intl/en/about/products?tab=wh"><u>More</u> &raquo;</a></nobr></div><div id=guser width=100%><nobr><span id=gbn class=gbi></span><span id=gbf class=gbf></span><span id=gbe></span><a href="http://www.google.com/history/optout?hl=en" class=gb4>Web History</a> | <a  href="/preferences?hl=en" class=gb4>Settings</a> | <a target=_top id=gb_70 href="https://accounts.google.com/ServiceLogin?hl=en&passive=true&continue=http://www.google.com/&ec=GAZAAQ" class=gb4>Sign in</a></nobr></div><div class=gbh style=left:0></div><div class=gbh style=right:0></div></div><center><br clear="all" id="lgpd"><div id="lga"><img alt="Google" height="92" src="/images/branding/googlelogo/1x/googlelogo_white_background_color_272x92dp.png" style="padding:28px 0 14px" width="272" id="hplogo"><br><br></div><form action="/search" name="f"><table cellpadding="0" cellspacing="0"><tr valign="top"><td width="25%">&nbsp;</td><td align="center" nowrap=""><input name="ie" value="ISO-8859-1" type="hidden"><input value="en" name="hl" type="hidden"><input name="source" type="hidden" value="hp"><input name="biw" type="hidden"><input name="bih" type="hidden"><div class="ds" style="height:32px;margin:4px 0"><input class="lst" style="margin:0;padding:5px 8px 0 6px;vertical-align:top;color:#000" autocomplete="off" value="" title="Google Search" maxlength="2048" name="q" size="57"></div><br style="line-height:0"><span class="ds"><span class="lsbb"><input class="lsb" value="Google Search" name="btnG" type="submit"></span></span><span class="ds"><span class="lsbb"><input class="lsb" id="tsuid_1" value="I\'m Feeling Lucky" name="btnI" type="submit"><script nonce="_ABgeM1p2MILLXMtY--z4A">(function(){var id=\'tsuid_1\';document.getElementById(id).onclick = function(){if (this.form.q.value){this.checked = 1;if (this.form.iflsig)this.form.iflsig.disabled = false;}\nelse top.location=\'/doodles/\';};})();</script><input value="AOEireoAAAAAZEakdZ-RmNiNq7C4Ntu-cnKMvtpnZmvL" name="iflsig" type="hidden"></span></span></td><td class="fl sblc" align="left" nowrap="" width="25%"><a href="/advanced_search?hl=en&amp;authuser=0">Advanced search</a></td></tr></table><input id="gbv" name="gbv" type="hidden" value="1"><script nonce="_ABgeM1p2MILLXMtY--z4A">(function(){var a,b="1";if(document&&document.getElementById)if("undefined"!=typeof XMLHttpRequest)b="2";else if("undefined"!=typeof ActiveXObject){var c,d,e=["MSXML2.XMLHTTP.6.0","MSXML2.XMLHTTP.3.0","MSXML2.XMLHTTP","Microsoft.XMLHTTP"];for(c=0;d=e[c++];)try{new ActiveXObject(d),b="2"}catch(h){}}a=b;if("2"==a&&-1==location.search.indexOf("&gbv=2")){var f=google.gbvu,g=document.getElementById("gbv");g&&(g.value=a);f&&window.setTimeout(function(){location.href=f},0)};}).call(this);</script></form><div id="gac_scont"></div><div style="font-size:83%;min-height:3.5em"><br><div id="prm"><style>.szppmdbYutt__middle-slot-promo{font-size:small;margin-bottom:32px}.szppmdbYutt__middle-slot-promo a.ZIeIlb{display:inline-block;text-decoration:none}.szppmdbYutt__middle-slot-promo img{border:none;margin-right:5px;vertical-align:middle}</style><div class="szppmdbYutt__middle-slot-promo" data-ved="0ahUKEwijv-Wi4cL-AhWylWoFHXw-CcQQnIcBCAQ"><a class="NKcBbd" href="https://www.google.com/url?q=https://artsandculture.google.com/experiment/zgFx1tMqeIZyTw%3Futm_source%3Dgoogle%26utm_medium%3Dhppromo%26utm_campaign%3Dcallinginourcorals&amp;source=hpp&amp;id=19034922&amp;ct=3&amp;usg=AOvVaw0nMWsnMoeASDuSYrKnPMNj&amp;sa=X&amp;ved=0ahUKEwijv-Wi4cL-AhWylWoFHXw-CcQQ8IcBCAU" rel="nofollow">Learn how to help restore coral reefs, simply by listening</a></div></div></div><span id="footer"><div style="font-size:10pt"><div style="margin:19px auto;text-align:center" id="WqQANb"><a href="/intl/en/ads/">Advertising</a><a href="/services/">Business Solutions</a><a href="/intl/en/about.html">About Google</a></div></div><p style="font-size:8pt;color:#70757a">&copy; 2023 - <a href="/intl/en/policies/privacy/">Privacy</a> - <a href="/intl/en/policies/terms/">Terms</a></p></span></center><script nonce="_ABgeM1p2MILLXMtY--z4A">(function(){window.google.cdo={height:757,width:1440};(function(){var a=window.innerWidth,b=window.innerHeight;if(!a||!b){var c=window.document,d="CSS1Compat"==c.compatMode?c.documentElement:c.body;a=d.clientWidth;b=d.clientHeight}a&&b&&(a!=google.cdo.width||b!=google.cdo.height)&&google.log("","","/client_204?&atyp=i&biw="+a+"&bih="+b+"&ei="+google.kEI);}).call(this);})();</script> <script nonce="_ABgeM1p2MILLXMtY--z4A">(function(){google.xjs={ck:\'xjs.hp.cZMjK1rN2dw.L.X.O\',cs:\'ACT90oGdWgvp7b-1i002ub8NTEiqmwqPag\',excm:[]};})();</script>  <script nonce="_ABgeM1p2MILLXMtY--z4A">(function(){var u=\'/xjs/_/js/k\\x3dxjs.hp.en.qkDX73W2TvU.O/am\\x3dAAAAOgEAFABY/d\\x3d1/ed\\x3d1/rs\\x3dACT90oFpp8_uyj9hwoAl3W3tvYwd1PFWOg/m\\x3dsb_he,d\';var amd=0;\nvar e=this||self,f=function(c){return c};var h;var n=function(c,g){this.g=g===l?c:""};n.prototype.toString=function(){return this.g+""};var l={};\nfunction p(){var c=u,g=function(){};google.lx=google.stvsc?g:function(){google.timers&&google.timers.load&&google.tick&&google.tick("load","xjsls");var a=document;var b="SCRIPT";"application/xhtml+xml"===a.contentType&&(b=b.toLowerCase());b=a.createElement(b);a=null===c?"null":void 0===c?"undefined":c;if(void 0===h){var d=null;var m=e.trustedTypes;if(m&&m.createPolicy){try{d=m.createPolicy("goog#html",{createHTML:f,createScript:f,createScriptURL:f})}catch(r){e.console&&e.console.error(r.message)}h=\nd}else h=d}a=(d=h)?d.createScriptURL(a):a;a=new n(a,l);b.src=a instanceof n&&a.constructor===n?a.g:"type_error:TrustedResourceUrl";var k,q;(k=(a=null==(q=(k=(b.ownerDocument&&b.ownerDocument.defaultView||window).document).querySelector)?void 0:q.call(k,"script[nonce]"))?a.nonce||a.getAttribute("nonce")||"":"")&&b.setAttribute("nonce",k);document.body.appendChild(b);google.psa=!0;google.lx=g};google.bx||google.lx()};google.xjsu=u;e._F_jsUrl=u;setTimeout(function(){0<amd?google.caft(function(){return p()},amd):p()},0);})();window._ = window._ || {};window._DumpException = _._DumpException = function(e){throw e;};window._s = window._s || {};_s._DumpException = _._DumpException;window._qs = window._qs || {};_qs._DumpException = _._DumpException;function _F_installCss(c){}\n(function(){google.jl={blt:\'none\',chnk:0,dw:false,dwu:true,emtn:0,end:0,ico:false,ikb:0,ine:false,injs:\'none\',injt:0,injth:0,injv2:false,lls:\'default\',pdt:0,rep:0,snet:true,strt:0,ubm:false,uwp:true};})();(function(){var pmc=\'{\\x22d\\x22:{},\\x22sb_he\\x22:{\\x22agen\\x22:true,\\x22cgen\\x22:true,\\x22client\\x22:\\x22heirloom-hp\\x22,\\x22dh\\x22:true,\\x22ds\\x22:\\x22\\x22,\\x22fl\\x22:true,\\x22host\\x22:\\x22google.com\\x22,\\x22jsonp\\x22:true,\\x22msgs\\x22:{\\x22cibl\\x22:\\x22Clear Search\\x22,\\x22dym\\x22:\\x22Did you mean:\\x22,\\x22lcky\\x22:\\x22I\\\\u0026#39;m Feeling Lucky\\x22,\\x22lml\\x22:\\x22Learn more\\x22,\\x22psrc\\x22:\\x22This search was removed from your \\\\u003Ca href\\x3d\\\\\\x22/history\\\\\\x22\\\\u003EWeb History\\\\u003C/a\\\\u003E\\x22,\\x22psrl\\x22:\\x22Remove\\x22,\\x22sbit\\x22:\\x22Search by image\\x22,\\x22srch\\x22:\\x22Google Search\\x22},\\x22ovr\\x22:{},\\x22pq\\x22:\\x22\\x22,\\x22rfs\\x22:[],\\x22sbas\\x22:\\x220 3px 8px 0 rgba(0,0,0,0.2),0 0 0 1px rgba(0,0,0,0.08)\\x22,\\x22stok\\x22:\\x22xbKthWzO6w9x03LgKKyCIggBEMM\\x22}}\';google.pmc=JSON.parse(pmc);})();</script>       </body></html>'


