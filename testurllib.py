import urllib3
http = urllib3.PoolManager()
r = http.request('GET', 'https://expired.badssl.com')
print(r.status)
print(r.data)
