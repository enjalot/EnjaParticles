/**
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 */


#ifndef _SRTS_RADIX_SORT_VERIFIER_H_
#define _SRTS_RADIX_SORT_VERIFIER_H_

#include <stdio.h>
#include <math.h>
#include <float.h>


template<typename T> 
void PrintValue(T val);

template<>
void PrintValue<short>(short val) {
	printf("%d", val);
}

template<>
void PrintValue<unsigned short>(unsigned short val) {
	printf("%d", val);
}

template<>
void PrintValue<float>(float val) {
	printf("%f", val);
}

template<>
void PrintValue<int>(int val) {
	printf("%d", val);
}

template<>
void PrintValue<unsigned int>(unsigned int val) {
	printf("%u", val);
}

template<>
void PrintValue<long>(long val) {
	printf("%ld", val);
}

template<>
void PrintValue<unsigned long>(unsigned long val) {
	printf("%lu", val);
}


template <typename T>
int VerifySort(T* sorted_keys, const unsigned int len, bool verbose) 
{
	
	for (int i = 1; i < len; i++) {

		if (sorted_keys[i] < sorted_keys[i - 1]) {
			printf("Incorrect: [%d]: ", i);
			PrintValue<T>(sorted_keys[i]);
			printf(" < ");
			PrintValue<T>(sorted_keys[i - 1]);

			if (verbose) {	
				printf("\n\n[...");
				for (int j = -4; j <= 4; j++) {
					if ((i + j >= 0) && (i + j < len)) {
						PrintValue<T>(sorted_keys[i + j]);
						printf(", ");
					}
				}
				printf("...]");
			}

			return 1;
		}
	}

	printf("Correct");
	return 0;
}



#endif
