import scrapy


class GianhaSpider(scrapy.Spider):
    name = 'gianha'
    start_urls = ['https://alonhadat.com.vn/nha-dat/can-ban/can-ho-chung-cu/ha-noi/413/quan-hoang-mai/trang--' + str(i) + '.html' for i in range(1,31)]
                    
    
    def parse(self, response):
        for apartment in response.css('div.content-item'):
            yield {
                'area': apartment.css('div.ct_dt::text').get(),
                'bedroom': apartment.css('span.bedroom::text').get(),
                'price': apartment.css('div.ct_price::text').get(),
                'floor': apartment.css('span.floors::text').get(),
                'road-width': apartment.css('span.road-width::text').get(),
                'parking': apartment.css('span.parking::text').get()
            } 
