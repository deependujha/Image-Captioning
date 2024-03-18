'use client';
import React from 'react';
import Image_Uploader from './Image_Uploader';

const My_Image_Caption_Card = () => {
	return (
		<div className="my-5 mx-5">
			<div
				className="border-2 border-black rounded-xl overflow-hidden"
				style={{
					width: '800px',
					height: '500px',
					backgroundColor: 'purple',
					display: 'flex',
					justifyContent: 'center',
					alignItems: 'center',
				}}
			>
				{/* <input type="file" accept='image/jpg, image/jpeg, image/png' /> */}
				<Image_Uploader />
			</div>
			
		</div>
	);
};

export default My_Image_Caption_Card;
