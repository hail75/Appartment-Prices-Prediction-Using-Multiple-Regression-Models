import scrapy


class GianhaSpider(scrapy.Spider):
    name = 'gianha'
    start_urls = ['https://alonhadat.com.vn/nha-dat/can-ban/can-ho-chung-cu/1/ha-noi/trang--' + str(i) + '.html' for i in range(1,41)]
    
    
    def parse(self, response):
        for apartment in response.css('div.content-item'):
            yield {
                'area': apartment.css('div.ct_dt::text').get(),
                'bedroom': apartment.css('span.bedroom::text').get(),
                'district': apartment.css('div.ct_dis a::text').getall()[2],
                'price': apartment.css('div.ct_price::text').get(),
            } 
