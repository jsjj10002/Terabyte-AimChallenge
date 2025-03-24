package com.sample.www.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PaginationDto {
	private int page;
	private int size;
	private int startPage;
	private int endPage;
	private int prePage;
	private int nextPage;
	private int totalPage;
}
