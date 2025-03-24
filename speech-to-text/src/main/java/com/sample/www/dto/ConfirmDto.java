package com.sample.www.dto;

import java.util.List;
import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
// 검사 dto
public class ConfirmDto {
	// result value
	private int resultCode;
	private String resultMassege;
}
