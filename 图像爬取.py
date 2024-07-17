import os
import requests
import jsonpath

def download_images(keyword, download_num):
    if not os.path.exists('Image-Downloader-master/phone dataset'):
        os.makedirs('Image-Downloader-master/phone dataset')
        print('新建 phone dataset 文件夹')

    if os.path.exists('phone dataset/'+keyword):
        print('文件夹 phone dataset/{} 已存在，之后直接将爬取到的图片保存至该文件夹中'.format(keyword))
    else:
        os.makedirs('phone dataset/{}'.format(keyword))
        print('新建文件夹：phone dataset/{}'.format(keyword))

    URL_1 = "https://image.baidu.com/search/acjson?tn=resultjson_com&logid=6747404891301413552&ipn=rj&ct=201326592&is=&fp=result&queryWord={}".format(keyword)
    URL_2 = "&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&word={}".format(keyword)
    page = 0
    URL_3 = "&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&expermode=&force=&cg=girl&pn={}".format(page)
    URL_4 = "&rn=30&gsm=96&1615640197716="
    URL = URL_1 + URL_2 + URL_3 + URL_4

    headers = {
                'sec-fetch-dest': 'image',
                'Host': 'image.baidu.com',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36 Edg/89.0.774.50',
                'Cookie': 'BIDUPSID=B3349D16232F3AD2AC467BAA6C8F70D0; PSTM=1584333213; indexPageSugList=%5B%22%E7%83%AD%E6%90%9C%E6%A6%9C%22%2C%22java%E8%A1%A8%E6%83%85%E5%8C%85%22%2C%22%E6%9F%AF%E5%B0%BC%E5%A1%9E%E6%A0%BC%E4%BB%AA%E8%A1%A8%E7%9B%98%22%2C%22%E5%A6%82%E4%BD%95%E6%9F%A5%E8%AF%A2%E7%9F%A5%E4%B9%8E%E4%BC%9A%E5%91%98%E5%86%85%E5%AE%B9%22%2C%22%E7%8C%AA%E5%85%AB%E6%88%92%22%2C%22%E5%86%99%E7%9C%9F%E7%85%A7%22%2C%22%E6%B3%B3%E8%A3%85%22%2C%22Django%20%E7%BB%B4%E6%8A%A4%E7%89%88%E6%9C%AC%22%2C%22TCP%2FIP%E5%8F%82%E8%80%83%E6%A8%A1%E5%9E%8B%22%5D; __yjs_duid=1_f55f800f1f606264f1477c9f550821561614323197826; BAIDUID=83265DD6C974B9C1FF6E639CD0277BC7:FG=1; MCITY=-236%3A; BAIDUID_BFESS=88F5383818F30474B77FAF4CFC91A277:FG=1; H_PS_PSSID=33512_33241_33256_33272_33594_33570_33392_26350; delPer=0; PSINO=3; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BA_HECTOR=a10lah012ha42h044p1g4pbkp0q; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; userFrom=null; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; ab_sr=1.0.0_MDQ0ZGIyODE1YTEyMmNmNjZkNzBmYmIxMzM4MjliY2Y5YjJjMDUxYWUxN2ZlNTBkNGNkZTUwNDI5N2UxMTM2MWFmNzZlMTJhOGM4YzUyMzI0YmRmYzZkOGFiZDdkMGMy',
            }

    response = requests.get(URL, headers=headers)
    if response.status_code == 200:
        print('爬取成功')
        img_urls = jsonpath.jsonpath(response.json(), '$..middleURL')

    i = 1
    download_set = set()
    page = 1
    FLAG = True

    def download_one_img(img_url, keyword, i=0):
        img_data_raw = requests.get(img_url).content
        file_name = i
        if 'f=JPEG' in img_url:
            postfix = '.jpg'
        elif 'f=PNG' in img_url:
            postfix = '.png'
        elif 'f=GIF' in img_url:
            postfix = '.gif'
        else:
            postfix = '.webp'
        img_path = '{}/img{}{}'.format(keyword, file_name, postfix)
        with open(img_path, 'wb') as f:
            f.write(img_data_raw)
            print('成功下载{}\n保存至 {}'.format(img_url, img_path))

    while FLAG:
        print('-------爬取第{}页-------'.format(page + 1))
        URL_3 = "&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&expermode=&force=&cg=girl&pn={}".format(page)
        URL = URL_1 + URL_2 + URL_3 + URL_4
        response = requests.get(URL, headers=headers)
        page += 1
        if response.status_code == 200:
            img_urls = jsonpath.jsonpath(response.json(), '$..middleURL')
            img_urls = set(img_urls) - download_set
            for img_url in img_urls:
                try:
                    download_one_img(img_url, keyword, i)
                    download_set.add(img_url)
                    i += 1
                    if i >= download_num:
                        print('爬取 {} 张图像完毕'.format(i))
                        FLAG = False
                        break
                except:
                    print('下载出错', img_url)

download_images('Apple iPhone 14', 200)
