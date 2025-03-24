package com.sample.www.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebMvcConfig implements WebMvcConfigurer {

	public static final String uploadImagesPath = "/home/ubuntu/tomcat/webapps/";
	// true 로컬 false 운영.
	public static final boolean isLocal = true;

	@Override
	public void addResourceHandlers(ResourceHandlerRegistry registry) {
		// registry.addResourceHandler("/js/**").addResourceLocations("classpath:/static/js/").setCachePeriod(2_592_000);
		// registry.addResourceHandler("/img/**").addResourceLocations("classpath:/static/img/").setCachePeriod(2_592_000);
		// registry.addResourceHandler("/font/**").addResourceLocations("classpath:/static/font/").setCachePeriod(2_592_000);
		// registry.addResourceHandler("/css/**").addResourceLocations("classpath:/static/css/").setCachePeriod(2_592_000);
		// registry.addResourceHandler("/media/**").addResourceLocations("classpath:/static/media/").setCachePeriod(2_592_000);
		if (!isLocal) {
			registry.addResourceHandler("/image/**").addResourceLocations("file:///" + uploadImagesPath)
					.setCachePeriod(604_800);
			registry.addResourceHandler("/video/**").addResourceLocations("file:///" + uploadImagesPath + "video/")
					.setCachePeriod(604_800);
			registry.addResourceHandler("/upload/**").addResourceLocations("file:///" + uploadImagesPath + "upload/")
					.setCachePeriod(604_800);
		}
	}
}